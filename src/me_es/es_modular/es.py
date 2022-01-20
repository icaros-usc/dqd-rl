# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------
#
# THIS IS NOT THE ORIGINAL VERSION OF THE FILE.
#
# Last modified 2021-12-02
"""Provides a class that handles work for an ME-ES individual."""

import logging
import time

import dask.distributed
import numpy as np

from src.objectives import GymControl, GymControlConfig
from src.objectives.obs_stats import ObsStats
from src.utils.worker_state import (get_noise_table, get_objective_module,
                                    init_objective_module)

from .logger import EvalResult, EvalStats, POResult, StepStats
from .optimizers import Adam, SimpleSGD
from .stats import batched_weighted_sum, compute_centered_ranks

# pylint: disable = missing-function-docstring, line-too-long, global-statement, logging-format-interpolation, too-many-instance-attributes, import-outside-toplevel

logger = logging.getLogger(__name__)


def initialize_worker(config: GymControlConfig):
    init_objective_module(GymControl, config)


def get_optim(args):
    if args['optimizer_args']['optimizer'] == 'sgd':
        return SimpleSGD
    if args['optimizer_args']['optimizer'] == 'adam':
        return Adam
    raise NotImplementedError


class ESIndividual:
    """Handles distributed rollouts for an individual theta.

    Two kinds of computations are supported. The first is computations for a
    gradient step -- in ME-ES, there are 10k of these. To execute this, call
    start_step() to get a list of promises and get_step() to collect results.
    The other kind of computation is evaluation of a single theta. To execute
    this, call start_theta_eval() and pass the futures returned into
    get_theta_eval().

    The actual job submission is handled in start_chunk() and get_chunk() --
    this is the main interface to the distributed library, whether Fiber (in the
    original code) or Dask (modified here).

    See the usage in es_manager.py for more info.
    """

    def __init__(self, pop_id, algo, args, client: dask.distributed.Client,
                 theta, bc, perf, bc_archive, noise_table):

        self.args = args
        self.algo = algo
        self.pop_id = pop_id
        self.theta = theta
        self.bc = bc
        self.perf = perf
        self.bc_archive = bc_archive
        self.noise_table = noise_table

        self.batches_per_step = self.args['num_workers']
        self.batch_size = self.args['batch_size']
        self.eval_batch_size = self.args['eval_batch_size']
        self.eval_batches_per_step = self.args['eval_batches_per_step']
        self.l2_coeff = self.args['optimizer_args']['l2_coeff']
        self.fitness_normalization = self.args['fitness_normalization']

        # fiber
        self.client = client

        logger.info('Population {} optimizing {} parameters'.format(
            pop_id, len(theta)))

        self.optimizer = get_optim(self.args)(
            self.theta, stepsize=self.args['optimizer_args']['learning_rate'])

        self.use_norm_obs = self.args['env_args']['use_norm_obs']

        # obs_stats
        self.obs_mean = None
        self.obs_std = None

        self.noise_std = self.args['noise_std']
        self.learning_rate = self.args['optimizer_args']['learning_rate']

        logger.info('Population {} created!'.format(pop_id))

    def start_chunk(self, runner, batches_per_step, batch_size, theta_future,
                    obs_stats_future, *args):
        """Starts batches_per_step jobs on the workers. Each job executes.

        batch_size evals, so we get batches_per_step * batch_size evals in
        total.

        Returns:
            A list of promises for completion of all the evals.
        """
        rs_seeds = np.random.randint(np.int32(2**31 - 1), size=batches_per_step)
        chunk_tasks = []
        for i in range(batches_per_step):
            chunk_tasks.append(
                self.client.submit(
                    runner,
                    batch_size,
                    rs_seeds[i],
                    theta_future,
                    obs_stats_future,
                    *args,
                    pure=False,
                ))
        return chunk_tasks

    def get_chunk(self, tasks):
        return self.client.gather(tasks)

    def collect_po_results(self, po_results):
        noise_inds = np.concatenate([r.noise_inds for r in po_results])
        returns = np.concatenate([r.returns for r in po_results])
        lengths = np.concatenate([r.lengths for r in po_results])
        bcs = np.concatenate([r.bcs for r in po_results])

        if not self.use_norm_obs:
            obs_updates = dict(obs_sums=None, obs_sqs=None, obs_counts=None)
        else:
            obs_sums = np.concatenate(
                [r.obs_sum.reshape(1, -1) for r in po_results], axis=0)
            obs_sqs = np.concatenate(
                [r.obs_sq.reshape(1, -1) for r in po_results], axis=0)
            obs_counts = np.array([r.obs_count for r in po_results])
            obs_updates = dict(obs_sums=obs_sums,
                               obs_sqs=obs_sqs,
                               obs_counts=obs_counts)
        return noise_inds, returns, lengths, bcs, obs_updates

    @staticmethod
    def collect_eval_results(eval_results):
        eval_returns = np.concatenate([r.returns for r in eval_results])
        eval_lengths = np.concatenate([r.lengths for r in eval_results])
        eval_bcs = np.concatenate([r.bcs for r in eval_results])
        eval_final_xpos = np.concatenate([r.final_xpos for r in eval_results])
        return eval_returns, eval_lengths, eval_bcs, eval_final_xpos

    def compute_grads(self, noise_inds, fitness, theta):
        grads, _ = batched_weighted_sum(
            fitness[:, 0] - fitness[:, 1],
            (self.noise_table.get(idx, len(theta)) for idx in noise_inds),
            batch_size=500)
        grads /= len(fitness)
        if self.args['optimizer_args']['divide_gradient_by_noise_std']:
            grads /= self.noise_std
        return grads

    def start_theta_eval(self, theta):
        """Submit tasks for evaluating one parameter."""
        logger.info('Starting evaluation of population {}.'.format(self.pop_id))
        theta_future = self.client.scatter(theta, broadcast=True)
        obs_stats_future = self.client.scatter((self.obs_mean, self.obs_std),
                                               broadcast=True)

        eval_task = self.start_chunk(
            run_eval_rollout_batch,
            self.eval_batches_per_step,
            self.eval_batch_size,
            theta_future,
            obs_stats_future,
        )
        return eval_task

    def get_theta_eval(self, new_theta, eval_task, start=False):
        eval_results = self.get_chunk(eval_task)
        eval_returns, eval_lengths, eval_bcs, eval_final_xpos = self.collect_eval_results(
            eval_results)

        if start:
            eval_novelties = 10 * np.ones(
                [3]
            )  # optimistic evaluation to run each population once at the beginning
        else:
            eval_novelties = self.compute_novelty(eval_bcs)

        logger.info(
            'Evaluation of best theta finished running {} episodes.'.format(
                len(eval_returns)))

        self.theta = new_theta.copy()
        self.perf = eval_returns.mean()
        self.bc = eval_bcs.mean(axis=0)

        return (
            eval_bcs,
            eval_returns,
            eval_novelties,
            eval_final_xpos,
            EvalStats(eval_returns_mean=eval_returns.mean(),
                      eval_returns_median=np.median(eval_returns),
                      eval_returns_std=eval_returns.std(),
                      eval_returns_max=eval_returns.max(),
                      eval_len_mean=eval_lengths.mean(),
                      eval_len_std=eval_lengths.std(),
                      eval_n_episodes=len(eval_returns) * len(eval_returns),
                      eval_novelty_mean=eval_novelties.mean(),
                      eval_novelty_median=np.median(eval_novelties),
                      eval_novelty_std=eval_novelties.std(),
                      eval_novelty_max=eval_novelties.max()),
        )

    def update_obs_stats(self, obs_stats):
        self.obs_mean = obs_stats.mean
        self.obs_std = obs_stats.std

    def start_step(self):
        """Start the jobs for calculating gradient step."""
        theta_future = self.client.scatter(self.theta, broadcast=True)
        obs_stats_future = self.client.scatter((self.obs_mean, self.obs_std),
                                               broadcast=True)

        training_task = self.start_chunk(run_po_rollout_batch,
                                         self.batches_per_step, self.batch_size,
                                         theta_future, obs_stats_future,
                                         self.noise_std)
        return training_task

    def get_step(self, training_task, explore, nsra_weight):
        """Collect job results from gradient step calculations."""
        step_results = self.get_chunk(training_task)

        noise_inds, po_returns, po_lengths, po_bcs, obs_updates = self.collect_po_results(
            step_results)

        episodes_this_step = po_returns.size
        timesteps_this_step = po_lengths.sum()
        logger.info(
            'Population {} finished running {} episodes, {} timesteps.'.format(
                self.pop_id, episodes_this_step, timesteps_this_step))

        # compute fitness
        po_proc_fitnesses, po_novelty = self.compute_fitness(
            returns=po_returns,
            bcs=po_bcs,
            explore=explore,
            nsra_weight=nsra_weight)

        # compute grads and update optimizer
        current_theta = self.theta.copy()
        grad = self.compute_grads(noise_inds, po_proc_fitnesses, current_theta)
        update_ratio, updated_theta = self.optimizer.update(
            current_theta, -grad + self.l2_coeff * current_theta)

        return updated_theta.copy(), self.optimizer, obs_updates, StepStats(
            po_returns_mean=po_returns.mean(),
            po_returns_median=np.median(po_returns),
            po_returns_std=po_returns.std(),
            po_returns_max=po_returns.max(),
            po_novelty_mean=po_novelty.mean(),
            po_novelty_std=po_novelty.std(),
            po_novelty_median=np.median(po_novelty),
            po_novelty_max=po_novelty.max(),
            po_returns_min=po_returns.min(),
            po_len_mean=po_lengths.mean(),
            po_len_std=po_lengths.std(),
            po_len_max=po_lengths.max(),
            noise_std=self.noise_std,
            learning_rate=self.optimizer.stepsize,
            theta_norm=np.square(self.theta).sum(),
            grad_norm=float(np.square(grad).sum()),
            update_ratio=float(update_ratio),
            episodes_this_step=episodes_this_step,
            timesteps_this_step=timesteps_this_step,
        )

    def compute_novelty(self, bcs):
        # compute novelty w.r.t. all theta previously encountered
        shape = bcs.shape
        dim_bc = shape[-1]
        bcs = bcs.reshape([np.prod(shape[:-1]), dim_bc])
        av_distance_to_knn = self.bc_archive.compute_novelty(bcs)
        novelty = av_distance_to_knn.reshape(shape[:-1])
        return novelty

    def compute_fitness(self, returns, bcs, explore, nsra_weight=0.5):
        novelty = self.compute_novelty(bcs)
        if self.algo == 'nses' or ('mees' in self.algo and explore):
            proc_fitness = self.normalize_fitness(novelty)
        elif self.algo == 'nsres':
            proc_fitness = (self.normalize_fitness(novelty) +
                            self.normalize_fitness(returns)) / 2.0
        elif self.algo == 'nsraes':
            proc_fitness = (1 - nsra_weight) * self.normalize_fitness(
                novelty) + nsra_weight * self.normalize_fitness(returns)
        elif 'mees' in self.algo and not explore:
            proc_fitness = self.normalize_fitness(returns)
        else:
            raise NotImplementedError
        return proc_fitness, novelty

    def normalize_fitness(self, fitness):
        if self.fitness_normalization == 'centered_ranks':
            proc_fitness = compute_centered_ranks(fitness)
        elif self.fitness_normalization == 'normal':
            proc_fitness = (fitness - fitness.mean()) / (fitness.std() + 1e-5)
        else:
            raise NotImplementedError(
                'Invalid return normalization `{}`'.format(
                    self.fitness_normalization))
        return proc_fitness


def run_po_rollout_batch(batch_size, rs_seed, theta, obs_stats, noise_std=None):
    noise_table = get_noise_table()
    t_init = time.time()
    obj_module = get_objective_module()
    obs_mean, obs_std = obs_stats
    rng = np.random.default_rng(rs_seed)

    assert noise_std is not None
    noise_inds = np.asarray(
        [noise_table.sample_index(rng, len(theta)) for _ in range(batch_size)],
        dtype='int')

    returns = np.zeros((batch_size, 2))
    final_xpos = np.zeros((batch_size, 2))
    lengths = np.zeros((batch_size, 2), dtype='int')
    bcs = [[None, None] for _ in range(batch_size)]

    # Mirror sampling - first direction.
    for i, new_theta in enumerate(
            theta + noise_std * noise_table.get(noise_idx, len(theta))
            for noise_idx in noise_inds):
        results = obj_module.evaluate(new_theta, 1, rng.integers(1e6),
                                      ObsStats(obs_mean, obs_std))
        returns[i, 0] = results.agg_return
        lengths[i, 0] = results.agg_length
        bcs[i][0] = results.agg_bc
        final_xpos[i, 0] = (np.nan if results.agg_final_xpos is None else
                            results.agg_final_xpos)

    # Mirror sampling - other direction. Collect obs stats on this direction.
    obs_sum = None
    obs_sumsq = None
    obs_count = None
    for i, new_theta in enumerate(
            theta - noise_std * noise_table.get(noise_idx, len(theta))
            for noise_idx in noise_inds):
        results = obj_module.evaluate(new_theta, 1, rng.integers(1e6),
                                      ObsStats(obs_mean, obs_std))
        returns[i, 1] = results.agg_return
        lengths[i, 1] = results.agg_length
        bcs[i][1] = results.agg_bc
        final_xpos[i, 1] = (np.nan if results.agg_final_xpos is None else
                            results.agg_final_xpos)
        obs_sum = results.obs_sum if obs_sum is None else obs_sum + results.obs_sum
        obs_sumsq = results.obs_sumsq if obs_sumsq is None else obs_sumsq + results.obs_sumsq
        obs_count = results.obs_count if obs_count is None else obs_count + results.obs_count

    end = time.time() - t_init
    return POResult(returns=returns,
                    noise_inds=noise_inds,
                    lengths=lengths,
                    bcs=np.array(bcs),
                    obs_sum=obs_sum,
                    obs_sq=obs_sumsq,
                    obs_count=obs_count,
                    time=end,
                    final_xpos=final_xpos)


def run_eval_rollout_batch(batch_size, rs_seed, theta, obs_stats):
    obj_module = get_objective_module()
    theta = theta.copy()
    obs_mean, obs_std = obs_stats
    res = obj_module.evaluate(theta,
                              batch_size,
                              rs_seed,
                              ObsStats(obs_mean, obs_std),
                              disable_noise=True)
    if res.final_xpos is None:
        res.final_xpos = np.full(batch_size, np.nan)
    return EvalResult(returns=res.returns,
                      lengths=res.lengths,
                      bcs=np.array(res.bcs).reshape(1, -1),
                      final_xpos=res.final_xpos)
