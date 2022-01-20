"""Runs robustness analysis for each logdir in the manifest.

This script is based on main.py. It should be invoked with the shell scripts
shown below since it requires distributed computation.

Example:
    bash scripts/run_robustness_local.sh manifest.yaml 42 8
"""
import itertools
import logging
from typing import List

import fire
import gin
import numpy as np
import pandas as pd
import torch
from dask.distributed import Client
from ribs.archives import ArchiveDataFrame

from src.analysis.figures import exclude_from_manifest, load_manifest
from src.analysis.utils import (is_me_es, load_experiment,
                                load_me_es_archive_df, load_metrics)
from src.main import check_env
from src.objectives import get_obj
from src.objectives.objective_result import ObjectiveResult
from src.objectives.obs_stats import ObsStats
from src.objectives.run_objective import run_objective
from src.utils.logging import setup_logging
from src.utils.worker_state import (close_objective_module_env,
                                    init_objective_module)


def robustness(client: Client, manifest: str, seed: int):
    """Calculates the robustness of all solutions in all logging directories.

    Robustness is calculated by evaluating each solution several times
    (configured with Manager.best_robustness) and comparing the mean objective
    to the one found during the experiment.

    Robustness results are saved next to the archive DF; e.g. if the archive was
    archive/archive_5000.pkl, the robustness is
    archive/archive_5000_robustness.pkl. This file contains a DF with rows
    corresponding to the rows of the archive DF.
    """
    paper_data, root_dir = load_manifest(manifest)

    for env in paper_data:
        for algo in paper_data[env]["algorithms"]:
            if exclude_from_manifest(paper_data, env, algo):
                continue

            logdirs = [d["dir"] for d in paper_data[env]["algorithms"][algo]]

            for logdir_path in logdirs:
                logging.info("----- Robustness for %s -----", logdir_path)
                logdir = load_experiment(root_dir / logdir_path)

                # Initialize objective modules everywhere. Copied from Manager.
                obj_name = gin.query_parameter("Manager.obj_name")
                config_class, objective_class = get_obj(obj_name)
                kwargs = {}
                if obj_name == "gym_control" and gin.query_parameter(
                        "Manager.use_norm_obs"):
                    use_norm_obs = True
                    kwargs["use_norm_obs"] = True
                else:
                    use_norm_obs = False
                config = config_class(**kwargs)
                # Note we use run() instead of register_worker_callbacks() here.
                # The lambda in register_worker_callbacks() may get called when
                # we are already processing the next logdir, which may be
                # troublesome.
                client.run(init_objective_module, objective_class, config)

                # Load archive data.
                total_gens = load_metrics(logdir).total_itrs
                logging.info("Loading archive data from gen %d", total_gens)
                archive_df: ArchiveDataFrame = (
                    load_me_es_archive_df(logdir)
                    if is_me_es() else pd.read_pickle(
                        logdir.file(f"archive/archive_{total_gens}.pkl")))
                logging.info("Solutions loaded: %d", len(archive_df))

                all_obs_stats = ([
                    ObsStats(m["obs_mean"], m["obs_std"])
                    for m in archive_df.batch_metadata()
                ] if use_norm_obs else itertools.repeat(None))

                # Only really necessary for ME-ES, the only algorithm which uses
                # action noise.
                eval_kwargs = {"disable_noise": True}

                # Use same number of evals for calculating robustness as was
                # used in experiment.
                logging.info("Evaluating solutions")
                n_evals = gin.query_parameter("Manager.best_robustness")
                futures = [
                    client.submit(
                        run_objective,
                        sol,
                        n_evals,
                        seed,  # Use the same base seed in all evals.
                        obs_stats,
                        eval_kwargs,
                        pure=False,
                    ) for sol, obs_stats in zip(archive_df.batch_solutions(),
                                                all_obs_stats)
                ]
                results: List[ObjectiveResult] = client.gather(futures)

                robustness_file = logdir.file(
                    f"archive/archive_{total_gens}_robustness.pkl")
                logging.info("Saving robustness results in %s", robustness_file)
                robustness_data = {
                    "agg_return": [r.agg_return for r in results],
                    "agg_bc": [r.agg_bc for r in results],
                    "agg_length": [r.agg_length for r in results],
                    "agg_final_xpos": [r.agg_final_xpos for r in results],
                    "returns": [r.returns for r in results],
                    "bcs": [r.bcs for r in results],
                    "lengths": [r.lengths for r in results],
                    "final_xpos": [r.final_xpos for r in results],
                }
                robustness_data["robustness"] = (
                    np.array(robustness_data["agg_return"]) -
                    archive_df.batch_objectives())
                pd.DataFrame(robustness_data).to_pickle(robustness_file)

                # To stop memory leaks.
                client.run(close_objective_module_env)

    logging.info("----- Done! -----")


def main(
    manifest: str,
    seed: int,
    address: str = "127.0.0.1:8786",
):
    """Parses command line flags and sets up and runs experiment.

    Args:
        manifest: Path to YAML file holding paths to logging directories. See
            `figures.py` for more info.
        seed: Master seed.
        address: Dask scheduler address.
    """
    check_env()

    client = Client(address)

    setup_logging(on_worker=False)
    client.register_worker_callbacks(setup_logging)

    client.register_worker_callbacks(lambda: torch.set_num_threads(1))

    logging.info("Waiting for at least 1 worker to join cluster")
    client.wait_for_workers(1)
    logging.info("At least one worker has joined")

    logging.info("Master Seed: %d", seed)
    logging.info("CPUs: %s", client.ncores())

    torch.manual_seed(seed + 42)

    robustness(client, manifest, seed)


if __name__ == "__main__":
    fire.Fire(main)
