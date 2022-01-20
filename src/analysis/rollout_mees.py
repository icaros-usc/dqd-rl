"""Visualize several rollouts of a policy from the ME-ES code.

This script should be run within a Singularity shell.

Usage:
    python -m src.analysis.rollout_me_es_policy LOGDIR MODEL.json

Example:
    python -m src.analysis.rollout_me_es_policy \
        logs/xyz/ logs/xyz/policies/0.json
"""
import json

import fire
import gin
import numpy as np
from alive_progress import alive_bar

from src.analysis.utils import load_experiment
from src.objectives import GymControl, GymControlConfig
from src.objectives.obs_stats import ObsStats


def main(logdir: str, model: str, n_evals: int = 5):
    """Evaluates the model for several iterations.

    Args:
        logdir: Logging directory associated with the model, used to retrieve
            configs.
        model: Path to JSON model.
        n_evals: Number of rollouts.
    """
    logdir = load_experiment(logdir)
    with open(model, "r") as file:
        data = json.load(file)
        theta = np.array(data["theta"], dtype=np.float32)
        obs_mean = np.array(data["obs_mean"], dtype=np.float32)
        obs_std = np.array(data["obs_std"], dtype=np.float32)

        for key in data:
            if key not in ("theta", "obs_mean", "obs_std"):
                print(f'"{key}": {data[key]}')

    use_norm_obs = gin.query_parameter("Manager.use_norm_obs")
    module = GymControl(
        GymControlConfig(
            use_norm_obs=use_norm_obs,
            render_delay=0.01,
        ))

    with alive_bar(n_evals, "Evals") as progress:
        for i in range(n_evals):
            res = module.evaluate(theta,
                                  1,
                                  i,
                                  ObsStats(obs_mean, obs_std),
                                  render=True,
                                  disable_noise=True)
            print(f"Objective: {res.returns[0]}\tBCs: {res.bcs[0]}\t"
                  f"Length: {res.lengths[0]}")
            progress()  # pylint: disable = not-callable


if __name__ == "__main__":
    fire.Fire(main)
