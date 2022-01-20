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
"""CONFIG_DEFAULT is a default configuration that can be run locally (although
it does not solve any problem as such).

To reproduce results from the 'Scaling Map-Elites to Deep Neuroevolution' paper,
the CONFIG_MEES_PAPER should be used.  It uses 1000 workers, which requires the
use of a cluster.

Link to the paper: https://arxiv.org/pdf/2003.01825.pdf

Note that the number of rollouts per generation is num_workers * batch_size * 2
(* 2  because of mirror sampling)
"""
import datetime
import json

CONFIG_DEFAULT = dict(
    n_iterations=1000,  # number of generations
    num_workers=2,  # number of workers
    # number of perturbations per worker per generation (each perturb leads to 2
    # rollouts because of mirror sampling).
    batch_size=1,
    eval_batches_per_step=2,  # number of rollouts for evaluation
    # number of evaluation rollout per worker. Leave to 1, as you have many more
    # workers than evaluation rollouts anyway
    eval_batch_size=1,
    # standard deviation of the Gaussian noise applied on the parameters for ES
    # updates
    noise_std=0.02,
    fitness_normalization='centered_ranks',
    # whether to use virtual batch normalization (computing running stats for
    # observations and normalizing the obs)
    env_args=dict(use_norm_obs=True),
    optimizer_args=dict(
        optimizer='adam',  # either adam or sgd
        learning_rate=0.01,
        l2_coeff=0.005,
        divide_gradient_by_noise_std=False
    ),  # OpenAI ES paper and POET use this, not sure why
    # Moved to gin.
    #  policy_args=dict(init='normc',
    #                   layers=[256, 256],
    #                   activation='tanh',
    #                   action_noise=0.01),
    novelty_args=dict(
        k=10,  # number of nearest neighbors for novelty search objective
        # 'nov' prob selects the pop proportionally to its novelty score, also
        # 'robin' (one after the other)
        nses_selection_method='nov_prob',
        nses_pop_size=5),  # number of populations
    mees_args=dict(
        # number of consecutive ES steps before choosing a new starting cell for
        # MEES
        nb_consecutive_steps=10,
        # method to bias starting cell sampling for exploration steps. Also
        # 'best_or_uniform'
        explore_strategy='novelty_bias',
        # method to bias starting cell sampling for exploration steps. Also
        # 'best_or_uniform'
        exploit_strategy='best2inlast5_or_best',
        strategy_explore_exploit='robin'
    )  # strategy for balancing between exploration and exploitation.
)

# Configuration for the damage recovery experiment
CONFIG_MEES_PAPER_DAMAGE = CONFIG_DEFAULT.copy()
CONFIG_MEES_PAPER_DAMAGE.update(
    env_id='DamageAnt-v2',
    n_iterations=500,
    # 1000 workers * 5 batch size * 2 (for mirror sampling) = 10000
    # (10k was stated in the paper, but it is not immediately obvious in the
    # code)
    num_workers=1000,
    batch_size=5,
    eval_batches_per_step=10,
    eval_batch_size=1)
CONFIG_MEES_PAPER_DAMAGE['mees_args'].update(explore_strategy='best_or_uniform',
                                             exploit_strategy='best_or_uniform')

# Configuration for the deep exploration experiment
CONFIG_MEES_PAPER_EXPLORATION = CONFIG_DEFAULT.copy()
CONFIG_MEES_PAPER_EXPLORATION.update(env_id='DeceptiveHumanoid-v2',
                                     n_iterations=1000,
                                     num_workers=1000,
                                     batch_size=5,
                                     eval_batches_per_step=10,
                                     eval_batch_size=1)
CONFIG_MEES_PAPER_EXPLORATION['mees_args'].update(
    explore_strategy='novelty_bias', exploit_strategy='best2inlast5_or_best')

# Custom configuration
CONFIG_CUSTOM = CONFIG_DEFAULT.copy()


def setup_config(args_dict):
    """Merges the given args into a config_dict and returns the result."""
    if args_dict['config'] == 'default':
        config_dict = CONFIG_DEFAULT
        config_dict.update(args_dict)
    elif args_dict['config'] == 'mees_damage':
        config_dict = CONFIG_MEES_PAPER_DAMAGE
        # We will use this for other environments.
        #  env_error_msg = (
        #       "The damage recovery experiment of the MEES paper was "
        #                   "conducted with the DamageAnt-v2 domain")
        #  assert args_dict['env_id'] == 'DamageAnt-v2', env_error_msg
    elif args_dict['config'] == 'mees_exploration':
        config_dict = CONFIG_MEES_PAPER_EXPLORATION
        # We will use this for other environments.
        #  env_error_msg = ("The deep exploration experiment of the MEES paper "
        #                   "was conducted with either 'HumanoidDeceptive-v2' "
        #                   "or 'AntMaze-v3'")
        #  assert args_dict['env_id'] in [
        #      'HumanoidDeceptive-v2',
        #      'AntMaze-v3',
        #  ], env_error_msg
        config_dict['env_id'] = args_dict['env_id']
    elif args_dict['config'] == 'custom':
        config_dict = CONFIG_CUSTOM
    else:
        raise NotImplementedError
    config_dict['algo'] = args_dict['algo']
    config_dict['time'] = str(datetime.datetime.now())  # Save job date/time.
    config_dict['log_dir'] = args_dict['log_dir']

    # Args passed into me_es_main.
    if args_dict["custom_n_iterations"] is not None:
        config_dict["n_iterations"] = args_dict["custom_n_iterations"]
    if args_dict["custom_nb_consecutive_steps"] is not None:
        config_dict["mees_args"]["nb_consecutive_steps"] = args_dict[
            "custom_nb_consecutive_steps"]
    if args_dict["custom_num_workers"] is not None:
        config_dict["num_workers"] = args_dict["custom_num_workers"]
    if args_dict["custom_batch_size"] is not None:
        config_dict["batch_size"] = args_dict["custom_batch_size"]
    if args_dict["custom_eval_batches_per_step"] is not None:
        config_dict["eval_batches_per_step"] = args_dict[
            "custom_eval_batches_per_step"]

    with open(config_dict['log_dir'] + 'config.json', 'wt') as file:
        json.dump(config_dict, file)

    return config_dict
