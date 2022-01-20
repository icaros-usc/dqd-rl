"""Functions for managing worker state.

In general, one uses these by first calling init_* or set_* to create the
attribute, then calling get_* to retrieve the corresponding value.
"""
from dask.distributed import get_worker

from src.objectives import ObjectiveBase
from src.utils.noise_table import NoiseTable

#
# Generic
#


def set_worker_state(key: str, val: object):
    """Sets worker_state[key] = val"""
    worker = get_worker()
    setattr(worker, key, val)


def get_worker_state(key: str) -> object:
    """Retrieves worker_state[key]"""
    worker = get_worker()
    return getattr(worker, key)


#
# Noise table
#

NOISE_TABLE_ATTR = "noise_table"


def init_noise_table():
    """Initializes this worker's noise table."""
    set_worker_state(NOISE_TABLE_ATTR, NoiseTable())


def get_noise_table() -> NoiseTable:
    """Retrieves this worker's copy of the noise table."""
    return get_worker_state(NOISE_TABLE_ATTR)


#
# Objective module
#

OBJECTIVE_MOD_ATTR = "objective_module"


def init_objective_module(module_class: ObjectiveBase, config: "config object"):
    """Initializes this worker's objective module."""
    set_worker_state(OBJECTIVE_MOD_ATTR, module_class(config))


def get_objective_module() -> ObjectiveBase:
    """Retrieves this worker's objective module."""
    return get_worker_state(OBJECTIVE_MOD_ATTR)

def close_objective_module_env() -> ObjectiveBase:
    """Closes the env in the objective module.

    Mainly for GymControl modules.
    """
    return get_worker_state(OBJECTIVE_MOD_ATTR).env.close()
