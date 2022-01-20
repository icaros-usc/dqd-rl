"""Provides a class for running each QD algorithm."""
import dataclasses
import itertools
import logging
import pickle as pkl
from typing import List, Tuple

import cloudpickle
import gin
import numpy as np
from dask.distributed import Client
from logdir import LogDir

from src.archives import GridArchive
from src.emitters import (GaussianEmitter, GradientImprovementEmitter,
                          IsoLineEmitter, PGAEmitter)
from src.objectives import actual_qd_score, get_analysis_id, get_obj
from src.objectives.gym_control.td3 import TD3, TD3Config
from src.objectives.objective_result import ObjectiveResult
from src.objectives.obs_stats import ObsStats, RunningObsStats
from src.objectives.run_objective import run_objective
from src.optimizers import Optimizer
from src.utils.deprecation import DEPRECATED
from src.utils.logging import worker_log
from src.utils.metric_logger import MetricLogger
from src.utils.worker_state import init_objective_module

# Just to get rid of pylint warning about unused import (adding a comment after
# each line above messes with formatting).
IMPORTS_FOR_GIN = (
    GaussianEmitter,
    IsoLineEmitter,
    GridArchive,
)

EMITTERS_WITH_RESTARTS = (GradientImprovementEmitter,)

logger = logging.getLogger(__name__)


@gin.configurable
class Manager:  # pylint: disable = too-many-instance-attributes
    """Runs a QD algorithm while handling distributed compute and logging.

    If you are trying to understand this code, first make sure you understand
    how the general pyribs loop works (https://pyribs.org). Essentially, the
    execute() method of this class runs this loop but in a more complicated
    fashion, as we want to distribute the solution evaluations, log various
    performance metrics, save various pieces of data, support reloading /
    checkpoints, etc. On top of that, we want to do this for a wide range of QD
    algorithms, such as PGA-ME, MAP-Elites, and CMA-MEGA-ES.

    Main args:
        client: Dask client for distributed compute.
        logdir: Directory for saving all logging info.
        seed: Master seed. The seed is not passed in via gin because it needs to
            be flexible.
        reload: If True, reload the experiment from the given logging directory.

    Algorithm args:
        archive_type: Archive class. Intended for gin configuration.
        emitter_types: List of tuples of (class, n); where each tuple indicates
            there should be n emitters with the given class. Intended for gin
            configuration.
        total_evals: Total number of (recorded) evaluations so use. This is used
            to determine the number of generations. To see how the evals per
            generation is calculated, see _calc_gens. Passing `generations`
            will override total_evals.
        generations: Total number of generations to run the algorithm.
        n_evals: Number of times to evaluate each solution.
        use_norm_obs: Whether to use observation normalization.
        is_dqd: Whether this algorithm is a DQD algorithm and thus should use
            gradients.
        use_td3: Whether this algorithms needs a TD3 object.
        call_jacobian: Whether to call jacobian_batch on the objective module.
            Otherwise, we handle the Jacobian calculations specially.
        experience_evals: Number of evals to collect experience for each
            solution. Only applicable when using TD3 (i.e. use_td3=True) with
            GymControl. Must be at most n_evals.

    Objective args:
        obj_name: Name of the objective function. See
            src.objectives.REGISTRY for allowable options.

    Logging args:
        archive_save_freq: Number of generations to wait before saving the full
            archive (i.e. including solutions and metadata). Set to None to
            never save (the archive will still be available in the reload file).
            Set to -1 to only save on the final iter.
        reload_save_freq: Number of generations to wait before saving
            reload data.
        save_results: Whether to save data on all returns and BCs
            (all_results.pkl). Frequency of saving is the same as
            reload_save_freq if this is True.
        plot_metrics_freq: Number of generations to wait before displaying text
            plot of metrics. Plotting is not expensive, but the output can be
            pretty large.
        best_robustness: Number of evals to use when calculating robustness of
            the best solution. Pass None to avoid calculating.
    """

    def __init__(
        self,
        ## Main args ##
        client: Client,
        logdir: LogDir,
        seed: int,
        reload: bool = False,
        ## Algorithm args ##
        archive_type=gin.REQUIRED,
        emitter_types: List[Tuple] = gin.REQUIRED,
        total_evals: int = None,
        generations: int = None,
        n_evals: int = gin.REQUIRED,
        use_norm_obs: bool = gin.REQUIRED,
        is_dqd: bool = gin.REQUIRED,
        use_td3: bool = gin.REQUIRED,
        call_jacobian: bool = gin.REQUIRED,
        experience_evals: int = gin.REQUIRED,
        ## Objective args ##
        obj_name: str = gin.REQUIRED,
        ## Logging args ##
        archive_save_freq: int = None,
        reload_save_freq: int = 5,
        save_results: bool = False,
        plot_metrics_freq: int = 5,
        best_robustness: int = None,
        ## Deprecated ##
        sample_gradients=DEPRECATED,  # pylint: disable = unused-argument
        small_archive_save_freq=DEPRECATED,  # pylint: disable = unused-argument
    ):  # pylint: disable = too-many-arguments, too-many-branches

        self._client = client
        self._logdir = logdir
        self._n_evals = n_evals
        self._experience_evals = experience_evals
        self._archive_save_freq = archive_save_freq
        self._reload_save_freq = reload_save_freq
        self._save_results = save_results
        self._plot_metrics_freq = plot_metrics_freq
        self._best_robustness = best_robustness
        self._use_norm_obs = use_norm_obs
        self._is_dqd = is_dqd
        self._use_td3 = use_td3
        self._call_jacobian = call_jacobian
        self._obj_name = obj_name
        self._analysis_id = get_analysis_id()

        # Set up an objective module locally and on workers. During evaluations,
        # run_objective retrieves this module and uses it to evaluate the
        # function. Configuration is done with gin (i.e. the params are in the
        # config file).
        config_class, objective_class = get_obj(obj_name)
        kwargs = {}
        if obj_name == "gym_control":
            kwargs["use_norm_obs"] = use_norm_obs
        config = config_class(**kwargs)
        self._objective_module = objective_class(config)
        client.register_worker_callbacks(
            lambda: init_objective_module(objective_class, config))

        # Temp storage for TD3 experience.
        self._td3_experience = []

        # The attributes below are either reloaded or created fresh. Attributes
        # added below must be added to the _save_reload_data() method.
        if not reload:
            logger.info("Setting up fresh components")
            self._rng = np.random.default_rng(seed)
            self._generations_completed = 0

            self._running_obs_stats = None
            if self._use_norm_obs:
                obs_shape = self._objective_module.obs_shape
                self._running_obs_stats = RunningObsStats(obs_shape)

            # pyribs archive.
            initial_solution = self._objective_module.initial_solution(seed)
            archive = archive_type(seed=seed, dtype=np.float32)
            logger.info("Archive: %s", archive)

            # pyribs emitters.
            emitters = []
            for emitter_class, n_emitters in emitter_types:
                emitter_seeds = self._rng.integers(np.iinfo(np.int32).max,
                                                   size=n_emitters,
                                                   endpoint=True)
                emitters.extend([
                    emitter_class(archive, initial_solution, seed=s)
                    for s in emitter_seeds
                ])
                logger.info("Constructed %d emitters of class %s - seeds %s",
                            n_emitters, emitter_class, emitter_seeds)
            logger.info("Emitters: %s", emitters)

            # pyribs optimizer.
            self._optimizer = Optimizer(archive, emitters)
            logger.info("Optimizer: %s", self._optimizer)

            # Set up TD3 module. We need TD3 to provide gradients for DQD for
            # the GymControl objective. We add some amount to the seed so that
            # we don't use the same seed everywhere.
            self._td3 = (TD3(
                TD3Config(),
                self._objective_module,
                self._optimizer.archive.behavior_dim,
                seed + 1337,
            ) if self._use_td3 else None)

            metric_list = [
                ("Total Evals", True),
                ("Mean Evaluation", False),
                *([
                    ("Mean Evaluation (Gradient)", False),
                ] if self._is_dqd else []),
                ("Actual QD Score", True),
                ("Archive Size", True),
                ("Archive Coverage", True),
                ("Best Performance", False),
                ("Worst Performance", False),
                ("Mean Performance", False),
                ("Overall Min Return", False),
                *([
                    ("Obs Mean L2 Norm", False),
                    ("Obs Std L2 Norm", False),
                    ("Obs Count", True),
                ] if self._use_norm_obs else []),
                *([
                    ("Total Restarts", True),
                ] if any(
                    isinstance(e, EMITTERS_WITH_RESTARTS)
                    for e in emitters) else []),
                *([
                    ("Replay Buffer Size", True),
                ] if self._use_td3 else []),
                *([
                    ("Robustness", False),
                ] if self._best_robustness is not None else []),
            ]
            for e_idx, e in enumerate(self._optimizer.emitters):
                if isinstance(e, PGAEmitter):
                    metric_list.append(
                        (f"Emitter {e_idx} - PGA Greedy Eval", False))
                if isinstance(e, GradientImprovementEmitter) and e.greedy_eval:
                    metric_list.append(
                        (f"Emitter {e_idx} - TD3 Greedy Performance", False))

            self._metrics = MetricLogger(metric_list)
            self._total_evals = 0
            self._overall_min_return = np.inf
            self._all_returns = []
            self._all_bcs = []

            self._metadata_id = 0
            self._cur_best_id = None  # ID of most recent best solution.
        else:
            logger.info("Reloading optimizer and other data from logdir")

            with open(self._logdir.pfile("reload.pkl"), "rb") as file:
                data = pkl.load(file)
                self._rng = data["rng"]
                self._generations_completed = data["generations_completed"]
                self._metrics = data["metrics"]
                self._running_obs_stats = data["running_obs_stats"]
                self._optimizer = data["optimizer"]
                self._total_evals = data["total_evals"]
                self._overall_min_return = data["overall_min_return"]
                self._all_returns = data["all_returns"]
                self._all_bcs = data["all_bcs"]
                self._metadata_id = data["metadata_id"]
                self._cur_best_id = data["cur_best_id"]

            logger.info("Generations already completed: %d",
                        self._generations_completed)
            logger.info("Execution continues from generation %d (1-based)",
                        self._generations_completed + 1)
            logger.info("Reloaded optimizer: %s", self._optimizer)

            if self._use_td3:
                self._td3 = TD3(
                    TD3Config(),
                    self._objective_module,
                    self._optimizer.archive.behavior_dim,
                    seed + 1337,
                ).load(
                    self._logdir.pfile("reload_td3.pkl"),
                    self._logdir.pfile("reload_td3.pth"),
                )
                logger.info("Reloaded TD3: %s", self._td3)
            else:
                self._td3 = None

        # Depends on reloaded variables.
        self._emitter_restarts = any(
            isinstance(e, EMITTERS_WITH_RESTARTS)
            for e in self._optimizer.emitters)

        self._calc_gens(generations, total_evals)

        logger.info("solution_dim: %d", self._optimizer.archive.solution_dim)

    def _calc_gens(self, generations, total_evals):
        """Calculates number of generations."""
        if generations is not None:
            self._generations = generations
            logger.info("Using pre-specified generations: %d",
                        self._generations)
        else:
            assert total_evals is not None, \
                "total_evals or generations must be provided"

            sols_per_gen = 0
            for e in self._optimizer.emitters:
                sols_per_gen += e.batch_size
                if isinstance(e, GradientImprovementEmitter):
                    sols_per_gen += 1 + e.sample_batch_size

            evals_per_gen = sols_per_gen * self._n_evals
            self._generations = total_evals // evals_per_gen

            if total_evals % evals_per_gen != 0:
                self._generations += 1
                logger.warning(
                    "total_evals not perfectly divisible by evals_per_gen; "
                    "we will use %d evals instead of %d",
                    self._generations * evals_per_gen,
                    total_evals,
                )

            logger.info("Using calculated generations: %d", self._generations)

    def _save_reload_data(self):
        """Saves data necessary for a reload.

        Current reload files:
        - reload.pkl
        - reload_td3.pth
        - reload_td3.pkl

        Since saving may fail due to memory issues, data is first placed in
        reload-tmp.pkl. reload-tmp.pkl then overwrites reload.pkl.

        We use gin to reference emitter classes, and pickle fails when dumping
        things constructed by gin, so we use cloudpickle instead. See
        https://github.com/google/gin-config/issues/8 for more info.
        """
        logger.info("Saving reload data")

        logger.info("Saving reload.pkl")
        with self._logdir.pfile("reload-tmp.pkl").open("wb") as file:
            cloudpickle.dump(
                {
                    "rng": self._rng,
                    "generations_completed": self._generations_completed,
                    "metrics": self._metrics,
                    "running_obs_stats": self._running_obs_stats,
                    "optimizer": self._optimizer,
                    "total_evals": self._total_evals,
                    "overall_min_return": self._overall_min_return,
                    "all_returns": self._all_returns,
                    "all_bcs": self._all_bcs,
                    "metadata_id": self._metadata_id,
                    "cur_best_id": self._cur_best_id,
                },
                file,
            )
        self._logdir.pfile("reload-tmp.pkl").rename(
            self._logdir.pfile("reload.pkl"))

        if self._use_td3:
            logger.info("Saving TD3 to reload_td3.pkl and reload_td3.pth")
            self._td3.save(
                self._logdir.pfile("reload_td3-tmp.pkl"),
                self._logdir.pfile("reload_td3-tmp.pth"),
            )
            self._logdir.pfile("reload_td3-tmp.pkl").rename(
                self._logdir.pfile("reload_td3.pkl"))
            self._logdir.pfile("reload_td3-tmp.pth").rename(
                self._logdir.pfile("reload_td3.pth"))

        logger.info("Finished saving reload data")

    def _save_result_data(self):
        """Saves _all_returns and _all_bcs to logdir/all_results.pkl.

        Though we cover this in the reload file, we also save it here because
        the reload file may not work later on due to API changes. This data is
        just lists and numpy arrays, so the pickle should be valid for a while.
        """
        self._logdir.save_data(
            {
                "all_returns": self._all_returns,
                "all_bcs": self._all_bcs,
            },
            "all_results-tmp.pkl",
        )
        self._logdir.pfile("all_results-tmp.pkl").rename(
            self._logdir.pfile("all_results.pkl"))

    def _save_archive(self):
        """Saves dataframes of the archive.

        The archive, including solutions and metadata, is saved to
        logdir/archive/archive_{generation}.pkl

        Note that the archive is saved as an ArchiveDataFrame storing common
        Python objects, so it should be stable (at least, given fixed software
        versions).
        """
        gen = self._generations_completed
        df = self._optimizer.archive.as_pandas(include_solutions=True,
                                               include_metadata=True)
        df.to_pickle(self._logdir.file(f"archive/archive_{gen}.pkl"))

    def _save_archive_history(self):
        """Saves the archive's history.

        We are okay with a pickle file here because there are only numpy arrays
        and Python objects, both of which are stable.
        """
        with self._logdir.pfile("archive_history.pkl").open("wb") as file:
            pkl.dump(self._optimizer.archive.history(), file)

    def _save_data(self):
        """Saves archive, reload data, history, and metrics if necessary.

        This method must be called at the _end_ of each loop. Otherwise, some
        things might not be complete. For instance, the metrics may be in the
        middle of an iteration, so when we reload, we get an error because we
        did not end the iteration.
        """
        is_final_gen = self._generations_completed == self._generations

        if self._archive_save_freq is None:
            save_full_archive = False
        elif self._archive_save_freq == -1 and is_final_gen:
            save_full_archive = True
        elif (self._archive_save_freq > 0 and
              self._generations_completed % self._archive_save_freq == 0):
            save_full_archive = True
        else:
            save_full_archive = False

        logger.info("Saving metrics")
        self._metrics.to_json(self._logdir.file("metrics.json"))

        if save_full_archive:
            logger.info("Saving full archive")
            self._save_archive()
        if ((self._generations_completed % self._reload_save_freq == 0) or
                is_final_gen):
            self._save_reload_data()
            if self._save_results:
                self._save_result_data()
        if is_final_gen:
            logger.info("Saving archive history")
            self._save_archive_history()

    def _plot_metrics(self):
        """Plots metrics every self._plot_metrics_freq gens or on final gen."""
        if (self._generations_completed % self._plot_metrics_freq == 0 or
                self._generations_completed == self._generations):
            logger.info("Metrics:\n%s", self._metrics.get_plot_text())

    def _update_result_stats(self, results: List[ObjectiveResult]):
        """Updates the overall min return and list of all returns."""
        result_min = np.min([np.min(r.returns) for r in results])
        self._overall_min_return = min(self._overall_min_return, result_min)

        for r in results:
            self._all_returns.append(r.returns)
            self._all_bcs.append(r.bcs)

    def _add_performance_metrics(self):
        """Calculates various performance metrics at the end of each iter."""
        df = self._optimizer.archive.as_pandas(include_solutions=False)
        objs = df.batch_objectives()
        stats = self._optimizer.archive.stats

        self._metrics.add(
            "Total Evals",
            self._total_evals,
            logger,
        )
        self._metrics.add(
            "Actual QD Score",
            actual_qd_score(objs, self._analysis_id),
            logger,
        )
        self._metrics.add(
            "Archive Size",
            stats.num_elites,
            logger,
        )
        self._metrics.add(
            "Archive Coverage",
            stats.coverage,
        )
        self._metrics.add(
            "Best Performance",
            np.max(objs),
            logger,
        )
        self._metrics.add(
            "Worst Performance",
            np.min(objs),
            logger,
        )
        self._metrics.add(
            "Mean Performance",
            np.mean(objs),
            logger,
        )
        self._metrics.add(
            "Overall Min Return",
            self._overall_min_return,
            logger,
        )
        if self._emitter_restarts:
            self._metrics.add(
                "Total Restarts",
                sum(e.restarts
                    for e in self._optimizer.emitters
                    if isinstance(e, EMITTERS_WITH_RESTARTS)),
                logger,
            )

        for e_idx, e in enumerate(self._optimizer.emitters):
            if isinstance(e, PGAEmitter):
                self._metrics.add(f"Emitter {e_idx} - PGA Greedy Eval",
                                  e.greedy_eval, logger)
            if isinstance(e, GradientImprovementEmitter) and e.greedy_eval:
                self._metrics.add(f"Emitter {e_idx} - TD3 Greedy Performance",
                                  e.greedy_obj, logger)

    def _extract_metadata(self, r: ObjectiveResult) -> dict:
        """Constructs metadata object from results of an evaluation."""
        meta = dataclasses.asdict(r)

        # Remove unwanted keys.
        none_keys = [key for key in meta if meta[key] is None]
        for key in itertools.chain(
                none_keys, ["obs_sum", "obs_sumsq", "obs_count", "experience"]):
            try:
                meta.pop(key)
            except KeyError:
                pass

        # Add observation stats.
        if self._use_norm_obs:
            meta.update({
                "obs_mean": self._running_obs_stats.mean.copy(),
                "obs_std": self._running_obs_stats.std.copy(),
                "obs_count": self._running_obs_stats.count,
            })

        meta["metadata_id"] = self._metadata_id
        self._metadata_id += 1

        return meta

    def _evaluate_solutions(self, with_gradients: bool):
        """Requests and evaluates solutions from the pyribs optimizer.

        Args:
            with_gradients: Whether to calculate jacobians (i.e. gradients).
        """
        logger.info("Requesting solutions from optimizer")
        logger.info("with_gradients=%s", with_gradients)

        ask_kwargs = []
        for e in self._optimizer.emitters:
            ask_kwargs.append(e_kwargs := {})
            if isinstance(e, GradientImprovementEmitter) and with_gradients:
                e_kwargs["grad_estimate"] = True
            if isinstance(e, (PGAEmitter, GradientImprovementEmitter)):
                e_kwargs["td3"] = self._td3
        solutions = self._optimizer.ask(ask_kwargs)
        logger.info("%d solutions generated", len(solutions))
        self._total_evals += self._n_evals * len(solutions)

        if self._use_norm_obs:
            logger.info("Broadcasting observation stats to workers.")
            logger.info("obs_mean: %s", self._running_obs_stats.mean)
            logger.info("obs_std: %s", self._running_obs_stats.std)
            obs_stats_future = self._client.scatter(
                ObsStats(self._running_obs_stats.mean,
                         self._running_obs_stats.std),
                broadcast=True,
            )
            if not with_gradients:
                self._metrics.add("Obs Mean L2 Norm",
                                  np.linalg.norm(self._running_obs_stats.mean),
                                  logger)
                self._metrics.add("Obs Std L2 Norm",
                                  np.linalg.norm(self._running_obs_stats.std),
                                  logger)
        else:
            obs_stats_future = None

        logger.info("Distributing evaluations")

        if len(solutions) == 1:
            # When there is only one solution, we distribute its evaluations
            # across all the workers so that the evaluation is faster. Each
            # worker evaluates the solution once.

            all_kwargs = [{} for _ in range(self._n_evals)]
            if self._use_td3:
                # `experience_evals` evaluations will collect experience.
                for i in range(self._experience_evals):
                    all_kwargs[i]["experience_evals"] = 1

            evaluation_seeds = self._rng.integers(np.iinfo(np.int32).max,
                                                  size=self._n_evals,
                                                  endpoint=True)
            futures = [
                self._client.submit(
                    run_objective,
                    solutions[0],
                    1,  # 1 eval per worker.
                    seed,
                    obs_stats_future,
                    eval_kwargs,
                    pure=False,
                ) for eval_kwargs, seed in zip(all_kwargs, evaluation_seeds)
            ]
        else:
            eval_kwargs = {
                "experience_evals": self._experience_evals
            } if self._use_td3 else {}

            # Make each solution evaluation have a different seed. Note that we
            # assign seeds to solutions rather than workers, which means that we
            # are agnostic to worker configuration.
            evaluation_seeds = self._rng.integers(np.iinfo(np.int32).max,
                                                  size=len(solutions),
                                                  endpoint=True)
            futures = [
                self._client.submit(
                    run_objective,
                    sol,
                    self._n_evals,
                    seed,
                    obs_stats_future,
                    eval_kwargs,
                    pure=False,
                ) for sol, seed in zip(solutions, evaluation_seeds)
            ]

        # We call train_critics() here so that it operates in parallel with the
        # with_gradients=False evaluations. We only train after one generation
        # is completed since there is no experience at 0 generations. _Since we
        # only train the critics here, the Jacobian in the first two generations
        # is random._ But this should not matter too much in the long run.
        if (self._use_td3 and not with_gradients and
                self._generations_completed >= 1):
            self._td3.train_critics()

        logger.info("Collecting evaluations")

        if len(solutions) == 1:
            single = self._client.gather(futures)  # List of ObjectiveResult.
            first = single[0]

            opts = None
            if self._obj_name == "gym_control":
                opts = gin.query_parameter("GymControlConfig.obj_result_opts")

            # Aggregate the results into one ObjectiveResult.
            results = [
                ObjectiveResult.from_raw(
                    returns=np.concatenate([r.returns for r in single]),
                    bcs=None if first.bcs is None else np.concatenate(
                        [r.bcs for r in single]),
                    lengths=None if first.lengths is None else np.concatenate(
                        [r.lengths for r in single]),
                    final_xpos=None if first.final_xpos is None else
                    np.concatenate([r.final_xpos for r in single]),
                    obs_sum=None if first.obs_sum is None else np.sum(
                        [r.obs_sum for r in single]),
                    obs_sumsq=None if first.obs_sumsq is None else np.sum(
                        [r.obs_sumsq for r in single]),
                    obs_count=None if first.obs_count is None else np.sum(
                        [r.obs_count for r in single]),
                    experience=None if first.experience is None else sum(
                        (r.experience for r in single), start=[]),
                    opts=opts,
                )
            ]
        else:
            results = self._client.gather(futures)  # List of ObjectiveResult.

        objs = [r.agg_return for r in results]
        bcs = [r.agg_bc for r in results]
        self._metrics.add(
            "Mean Evaluation (Gradient)"
            if with_gradients else "Mean Evaluation",
            np.mean(objs),
            logger,
        )
        self._update_result_stats(results)

        logger.info("Collecting metadata")
        metadata = [self._extract_metadata(r) for r in results]

        if with_gradients:
            logger.info("Calculating Jacobians")
            if self._call_jacobian:
                jacobians = self._objective_module.jacobian_batch(solutions)
            else:
                jacobians = None  # Emitter figures it out in this case.
        else:
            jacobians = None

        logger.info("Returning results to optimizer")

        # This tends to make the scheduler output very large because there is so
        # much info here.
        #  logger.info("Objs:\n%s", objs)
        #  logger.info("BCs:\n%s", bcs)

        tell_kwargs = []
        for e in self._optimizer.emitters:
            tell_kwargs.append(e_kwargs := {})
            if isinstance(e, GradientImprovementEmitter) and with_gradients:
                e_kwargs["grad_estimate"] = True
                e_kwargs["td3"] = self._td3

        self._optimizer.tell(objs, bcs, metadata, jacobians, tell_kwargs)

        if self._use_norm_obs:
            logger.info("Updating observation stats")
            obs_count = np.sum([r.obs_count for r in results])
            if not with_gradients:
                self._metrics.add("Obs Count", obs_count, logger)
            obs_mean_diff, obs_std_diff = self._running_obs_stats.increment(
                np.sum([r.obs_sum for r in results], axis=0),
                np.sum([r.obs_sumsq for r in results], axis=0),
                obs_count,
            )
            logger.info("obs_mean_diff: %s", obs_mean_diff)
            logger.info("obs_std_diff: %s", obs_std_diff)

        if self._use_td3:
            for r in results:
                self._td3_experience.append(r.experience)

    def _calc_best_robustness(self):
        """Calculates robustness of the best solution in the archive.

        Robustness is calculated as the difference between the mean evaluation
        of the top solution and its current evaluation in the archive.

        The difference is usually negative, and the higher it is the better.
        """
        if self._best_robustness is None:
            logger.info("Robustness evaluation not requested")
            return

        logger.info("Calculating robustness of best solution")
        elite = self._optimizer.archive.best_elite()

        if (self._cur_best_id is not None and
                elite.meta["metadata_id"] == self._cur_best_id):
            logger.info("Best elite has not changed - skipping robustness calc")
            prev_robustness = self._metrics.get_single("Robustness")["y"][-1]
            self._metrics.add("Robustness", prev_robustness, logger)
            return

        self._cur_best_id = elite.meta["metadata_id"]

        if self._use_norm_obs:
            logger.info("Broadcasting observation stats to workers.")
            logger.info("obs_mean: %s", elite.meta["obs_mean"])
            logger.info("obs_std: %s", elite.meta["obs_std"])
            obs_stats_future = self._client.scatter(
                ObsStats(elite.meta["obs_mean"], elite.meta["obs_std"]),
                broadcast=True,
            )
        else:
            obs_stats_future = None

        logger.info("Distributing robustness evals - %d evals",
                    self._best_robustness)

        evaluation_seeds = self._rng.integers(np.iinfo(np.int32).max,
                                              size=self._best_robustness,
                                              endpoint=True)
        futures = [
            self._client.submit(
                run_objective,
                elite.sol,
                1,  # 1 eval per worker.
                seed,
                obs_stats_future,
                {},  # eval_kwargs
                pure=False,
            ) for seed in evaluation_seeds
        ]

        logger.info("Collecting evaluations")
        results = self._client.gather(futures)
        mean_obj = np.mean([r.agg_return for r in results])
        robustness = mean_obj - elite.obj
        self._metrics.add("Robustness", robustness, logger)

    def execute(self):
        """Runs the entire algorithm."""
        for generation in range(self._generations_completed + 1,
                                self._generations + 1):
            logger.info(msg := (f"---------- Generation {generation}/"
                                f"{self._generations} ----------"))
            self._client.run(worker_log, msg)
            self._metrics.start_itr()
            self._optimizer.archive.new_history_gen()

            # Evaluation with gradients -> only for DQD algorithms.
            if self._is_dqd:
                self._evaluate_solutions(with_gradients=True)

            # Evaluate solutions without gradients -> applies to all algorithms.
            self._evaluate_solutions(with_gradients=False)

            # Add in all the TD3 experience.
            if self._use_td3:
                for e in self._td3_experience:
                    self._td3.add_experience(e)
                self._td3_experience.clear()  # Clean up for next itr.
                self._metrics.add("Replay Buffer Size", len(self._td3.buffer))

            self._calc_best_robustness()

            logger.info("Generation complete - now logging and saving data")
            self._generations_completed += 1
            self._add_performance_metrics()
            self._metrics.end_itr()
            self._plot_metrics()
            self._save_data()  # Keep at end of loop (see method docstring).
        logger.info("---------- Done! ----------")
