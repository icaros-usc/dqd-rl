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
"""(ME-ES) MAIN SCRIPT.

The default parameters are set in config.py The argument of this script override
the default configuration.
"""
import logging

import dask.distributed
import gin
import numpy as np

from .config import setup_config
from .es_modular.es_manager import ESPopulationManager

logger = logging.getLogger(__name__)


@gin.configurable(denylist=["log_dir"])
def me_es_main(
    client: dask.distributed.Client,
    log_dir: str,
    seed: int,
    algo: str = "mees_explore_exploit",
    config: str = "default",
    n_iterations: int = None,
    nb_consecutive_steps: int = None,
    num_workers: int = None,
    batch_size: int = None,
    eval_batches_per_step: int = None,
):
    """Runs the me-es paper code.

    Args:
        log_dir: logging directory
        algo: 'mees_explore_exploit', 'mees_explore', 'mees_exploit'
        env_id: 'AntMaze-v3', 'HumanoidDeceptive-v2' or 'DamageAnt-v2'
        seed: Seed for random number generators.
        config: 'default', 'mees_damage', 'mees_exploration' or 'custom'
    """
    args_dict = {
        "log_dir": log_dir,
        "algo": algo,
        "env_id": gin.query_parameter("GymControlConfig.env_id"),
        "config": config,
        "custom_n_iterations": n_iterations,
        "custom_nb_consecutive_steps": nb_consecutive_steps,
        "custom_num_workers": num_workers,
        "custom_batch_size": batch_size,
        "custom_eval_batches_per_step": eval_batches_per_step,
    }

    # Override default configuration with arguments.
    # Params are stored into a dict and saved in log_dir.
    config = setup_config(args_dict)

    logger.info(config)  # Print config in logs.
    np.random.seed(seed)  # Set master seed.

    # Run optimization.
    optimizer = ESPopulationManager(client, args=config)
    optimizer.optimize(iterations=config['n_iterations'])
