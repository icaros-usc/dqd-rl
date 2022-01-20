"""Objective functions and associated utilities.

Objective functions are implemented as "objective modules" which are child
classes of ObjectiveBase. This init file exposes the currently available
objectives and provides utilities like the REGISTRY, ANALYSIS_INFO, and
actual_qd_score.

When adding an objective module, make sure to update REGISTRY and ANALYSIS_INFO.

IMPORTANT: Avoid logging large amounts of data in objective modules, as we
need solution evaluations to be fast.
"""

import warnings
from typing import Optional

import gin
import numpy as np

from src.objectives.gym_control import GymControl, GymControlConfig
from src.objectives.objective_base import ObjectiveBase

# Mapping from objective names to configuration class and objective class.
REGISTRY = {
    "gym_control": (GymControlConfig, GymControl),
}


def get_obj(name: str) -> ("config class", "objective class"):
    """Retrieves the config and objective class associated with the given name.

    See REGISTRY for all acceptable names.
    """
    return REGISTRY[name]


# Information about the objectives that is useful in analysis.
#
# Min score is taken from the experiments run for our paper. Max score is taken
# from the reward thresholds here:
# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/__init__.py
ANALYSIS_INFO = {
    "QDAntBulletEnv-v0": {
        "min_score": -374.702526595403,
        "max_score": 2500.0,
    },
    "QDHalfCheetahBulletEnv-v0": {
        "min_score": -2797.5195251991777,
        "max_score": 3000.0,
    },
    "QDHopperBulletEnv-v0": {
        "min_score": -362.0897231940748,
        "max_score": 2500.0,
    },
    "QDWalker2DBulletEnv-v0": {
        "min_score": -67.16652198025106,
        "max_score": 2500.0,
    },
}


def get_analysis_id() -> str:
    """Uses gin to retrieve an ID for analyzing the objective.

    This analysis ID is used to access data in ANALYSIS_INFO.
    """
    obj_name = gin.query_parameter("Manager.obj_name")
    return (gin.query_parameter("GymControlConfig.env_id")
            if obj_name == "gym_control" else obj_name)


def actual_qd_score(objs: "array-like", analysis_id: Optional[str] = None):
    """Calculates QD score of the archive in the given env.

    Scores are normalized to be positive by subtracting a constant min score
    based on the environment (this ensures scores are computed the same way
    across environments).

    Args:
        objs: List of objective values.
        analysis_id: See get_analysis_id. If not passed in, we will call
            get_analysis_id().
    """
    analysis_id = get_analysis_id() if analysis_id is None else analysis_id
    objs = np.array(objs)
    objs -= ANALYSIS_INFO[analysis_id]["min_score"]
    if np.any(objs < 0):
        warnings.warn("Some objective values are still negative.")
    return np.sum(objs)
