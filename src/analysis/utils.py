"""Utilities for other postprocessing scripts.

Note that most of these functions require that you first call `load_experiment`
so that gin configurations are loaded properly.
"""
import json
import pickle as pkl
from pathlib import Path
from typing import Union

import gin
import numpy as np
import pandas as pd
from logdir import LogDir
from ribs.archives import ArchiveDataFrame

# Including this makes gin config work because main imports (pretty much)
# everything.
import src.main  # pylint: disable = unused-import
from src.archives import GridArchive
from src.objectives import ANALYSIS_INFO, get_analysis_id
from src.utils.deprecation import DEPRECATED_OBJECTS
from src.utils.metric_logger import MetricLogger


def load_experiment(logdir: str) -> LogDir:
    """Loads gin configuration and logdir for an experiment.

    Intended to be called at the beginning of an analysis script.

    Args:
        logdir: Path to the experiment's logging directory.
    Returns:
        LogDir object for the directory.
    """
    gin.clear_config()  # Erase all previous param settings.
    gin.parse_config_file(Path(logdir) / "config.gin",
                          skip_unknown=DEPRECATED_OBJECTS)
    logdir = LogDir(gin.query_parameter("experiment.name"), custom_dir=logdir)
    return logdir


def is_me_es() -> bool:
    """Indicates whether this experiment uses the original ME-ES code."""
    try:
        return gin.query_parameter("experiment.use_me_es")
    except ValueError:
        return False


def load_me_es_metrics(logdir: LogDir) -> MetricLogger:
    """Retrieves metrics from an ME-ES logging dir.

    Only certain metrics are loaded.

    Make sure that the gin config for the logdir has already been loaded with
    load_experiment().
    """
    data = pd.read_csv(logdir.file("results.csv"))
    analysis_id = get_analysis_id()
    max_archive_size = np.product(gin.query_parameter("GridArchive.dims"))

    metrics = MetricLogger([
        ("Total Evals", True),
        ("Actual QD Score", True),
        ("Archive Size", True),
        ("Archive Coverage", True),
        ("Best Performance", False),
        ("Mean Performance", False),
        ("Robustness", False),
    ])

    for i, (total_evals, qd_score, archive_size, best_perf,
            robustness) in enumerate(
                zip(data["episodes_so_far"], data["qd_score"],
                    data["archive_size"],
                    data["overall_best_eval_returns_mean"],
                    data["robustness"])):
        metrics.start_itr()

        mean_perf = (qd_score + archive_size *
                     ANALYSIS_INFO[analysis_id]["min_score"]) / archive_size

        metrics.add("Total Evals", total_evals)
        metrics.add("Actual QD Score", qd_score)
        metrics.add("Archive Size", archive_size)
        metrics.add("Archive Coverage", archive_size / max_archive_size)
        metrics.add("Best Performance", best_perf)
        metrics.add("Mean Performance", mean_perf)
        metrics.add("Robustness", robustness)

        itr_time = (data["time_elapsed_so_far"][i]
                    if i == 0 else data["time_elapsed_so_far"][i] -
                    data["time_elapsed_so_far"][i - 1])
        metrics.end_itr(itr_time)

    return metrics


def load_metrics(logdir) -> MetricLogger:
    return (load_me_es_metrics(logdir) if is_me_es() else
            MetricLogger.from_json(logdir.file("metrics.json")))


def load_me_es_objs(
    logdir: LogDir,
    gen: int = None,
    return_archive: bool = False,
) -> Union[np.ndarray, GridArchive]:
    """Retrieves objectives at the given gen from an ME-ES logging dir.

    If gen is None, the last generation is loaded.

    If return_archive is True, we instead return a "dummy" archive with all the
    objectives.
    """
    if gen is None:
        gen = gin.query_parameter("me_es_main.n_iterations")

    # Each history entry looks like:
    # [action, str_history, bc_update, perf_update, iter, cell_id, starting_bc,
    # explore]
    with logdir.pfile("archive/history.pk").open("rb") as file:
        history = pkl.load(file)

    # Use a "dummy" archive to keep track of objectives as we replay history.
    archive = GridArchive(  # pylint: disable = no-value-for-parameter
        dtype=np.float32)
    archive.initialize(0)  # No need to track any solutions.

    for h in history[:gen + 1]:  # Add 1 to account for initial sol of ME-ES.
        archive.new_history_gen()
        if h[0] != "nothing":
            archive.add([], h[3], h[2], None)

    return archive if return_archive else archive.as_pandas(
        include_solutions=False).batch_objectives()


def load_me_es_archive_df(logdir: LogDir) -> ArchiveDataFrame:
    """Loads the archive from the most recent data in the ME-ES directory.

    WARNING: This ArchiveDataFrame currently does not have any behaviors or
    indices, though these could be added in the future.
    """
    data = {}
    cell_ids = np.loadtxt(
        logdir.file("archive/final_filled_cells.txt")).astype(int)

    data["objective"] = np.loadtxt(logdir.file("archive/final_me_perfs.txt"))

    solutions, metadata = [], []
    for cell_id in cell_ids:
        with open(logdir.file(f"policies/{cell_id}.json", "r")) as file:
            policy = json.load(file)
            solution = np.array(policy["theta"], dtype=np.float32)
            obs_mean = np.array(policy["obs_mean"], dtype=np.float32)
            obs_std = np.array(policy["obs_std"], dtype=np.float32)

        solutions.append(solution)
        metadata.append({"obs_mean": obs_mean, "obs_std": obs_std})

    solutions = np.array(solutions)
    for i in range(len(solutions[0])):
        data[f"solution_{i}"] = solutions[:, i]

    data["metadata"] = metadata

    return ArchiveDataFrame(data, copy=False)


def load_archive_from_history(logdir: LogDir) -> GridArchive:
    """Generator that produces archives loaded from archive_history.pkl.

    Note that these archives will only contain objectives and BCs.
    """
    archive_type = str(gin.query_parameter("Manager.archive_type"))
    if archive_type == "@GridArchive":
        # Same construction as in Manager.
        # pylint: disable = no-value-for-parameter
        archive = GridArchive(seed=42, dtype=np.float32)
    else:
        raise TypeError(f"Cannot handle archive type {archive_type}")
    archive.initialize(0)  # No solutions.

    with logdir.pfile("archive_history.pkl").open("rb") as file:
        archive_history = pkl.load(file)

    yield archive  # Start with empty archive.
    for gen_history in archive_history:
        archive.new_history_gen()
        for obj, bcs in gen_history:
            archive.add([], obj, bcs, None)  # No solutions, no metadata.
        yield archive


def load_archive_gen(logdir: LogDir, gen: int) -> GridArchive:
    """Loads the archive at a given generation; works for ME-ES too."""
    if is_me_es():
        return load_me_es_objs(logdir, gen, return_archive=True)
    else:
        itr = iter(load_archive_from_history(logdir))
        for _ in range(gen + 1):
            archive = next(itr)
        return archive
