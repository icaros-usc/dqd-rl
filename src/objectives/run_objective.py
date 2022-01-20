"""Provides a generic function for executing objective functions."""
import logging
import time
from typing import Union

import numpy as np

from src.objectives.objective_result import ObjectiveResult
from src.objectives.obs_stats import ObsStats
from src.utils.noise_table import NoiseVector
from src.utils.worker_state import get_objective_module

logger = logging.getLogger(__name__)


def run_objective(solution: Union[np.ndarray, NoiseVector],
                  n_evals: int,
                  seed: int,
                  obs_stats: ObsStats = None,
                  eval_kwargs=None) -> ObjectiveResult:
    """Grabs the objective module and evaluates solution n_evals times.

    The objective module on this worker should have an evaluate() function that
    supports one of two signatures. If obs_stats is not None, evaluate() should
    take in solution, n_evals, seed, and obs_stats. Otherwise, evaluate() should
    take in solution, n_evals, and seed.
    """
    start = time.time()
    logger.info("run_objective with %d n_evals and seed %d", n_evals, seed)
    objective_module = get_objective_module()
    eval_kwargs = {} if eval_kwargs is None else eval_kwargs

    if obs_stats is not None:
        logger.info("Obs Mean L2 norm on worker: %f",
                    np.linalg.norm(obs_stats.mean))
        logger.info("Obs Std L2 norm on worker: %f",
                    np.linalg.norm(obs_stats.std))
        result = objective_module.evaluate(solution, n_evals, seed, obs_stats,
                                           **eval_kwargs)
    else:
        result = objective_module.evaluate(solution, n_evals, seed,
                                           **eval_kwargs)

    # This is a BAD IDEA particularly when collecting experience - logging such
    # a large result greatly slows down everything.
    #  logger.info("Result: %s", result)

    logger.info("run_objective done after %f sec", time.time() - start)

    return result
