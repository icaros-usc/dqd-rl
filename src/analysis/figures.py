r"""Plots figures and tables for the paper.

## Overview

The experiments output logging directories which are large and difficult to
manage. This script first gathers the relevant data from these directories into
one file, `figure_data.json`. `figure_data.json` is then passed around in order
to make the figures.

## Generating `figure_data.json` from logging directories

(If you already have `figure_data.json`, skip this section)

After running your experiments, arrange your logging directories as follows:

    logs/  # Any name is okay; you could even put it in `.` but that is messy.
      - manifest.yaml
      - logging-directory-1
      - logging-directory-2
      - ...

`manifest.yaml` lists the directories that were generated from your experiments.
It must be located in the same directory as all the logging directories, and it
must have the following format:

    Paper:  # Top-level object.
      Environment 1:
        old_min_obj: -8765.4321...
        min_obj: -1234.5678...
        max_obj: 3210.5678...
        archive_size: 1024
        algorithms:
            Algorithm 1:
              - dir: logging-directory-1...
                seed: 1
              - dir: logging-directory-2...
                seed: 3  # Make sure this matches the seed for the experiment.
              ...
            Algorithm 2:
              - exclude  # Causes this algorithm to be excluded.
              - dir: logging-directory-3...
                seed: 1
              ...
            ...
      ...

The fields are as follows:
- `old_min_obj` and `min_obj` are used for the QD score calculations --
  `old_min_obj` is the min that was used for calculating the QD score during the
  experiment, and after the experiment, we recalculate the QD score with the
  `min_obj`. This is necessary since the final QD score offset is based on the
  lowest objective value that was ever inserted into the archive (see the
  `find_min` function below), and we do not know this value during the
  experiments.
- `max_obj` is the maximum objective in the environment
- `archive_size` is the number of cells in the archive grid

You can leave `min_obj` blank for now. We'll generate it in the next step.

Once you have this manifest, run the following commands (replace
`logs/manifest.yaml` with the path to your manifest). Run all Python commands in
the Singularity container associated with this project, e.g. run `make shell`
to start a shell in the container and run the commands in that shell.

    # Collect min objectives for each environment with the following command,
    # and manually add these under the min_obj field in the manifest.
    python -m src.analysis.figures find_min logs/manifest.yaml

    # Run robustness calculations. See analysis/robustness.py for more details.
    # This command will take a while and probably requires a multi-core machine.
    bash scripts/run_robustness_local.sh logs/manifest.yaml 42 8

    # Generate `figure_data.json`
    python -m src.analysis.figures collect logs/manifest.yaml

For reference, figure_data.json looks like this:

    {
        "Env 1": {
            # List of entries for the algorithm, where each entry contains data
            # from a logging directory.
            "Algo 1": [
                {
                    # Number of evaluations completed after each iteration. Used
                    # on x-axis.
                    "Evaluations": [...],

                    # Metrics with a series of values from each generation. Some
                    # lists have length `gens + 1` because they include a 0 at
                    # the start, and others only have length `gens`.
                    "series_metrics": {
                        "QD Score": [...],

                        # QD Score divided by max QD score. Only used in
                        # statistical analysis.
                        "Normalized QD Score": [...],

                        "Archive Coverage": [...],

                        "Best Performance": [...],
                    }

                    # Metrics that only have a final value.
                    "point_metrics": {
                        # Average robustness of elites in final archive.
                        "Mean Elite Robustness": XXX,

                        # Mean Elite Robustness divided by objective range (max_obj -
                        # min_obj). Only used in statistical analysis.
                        "Normalized Mean Elite Robustness": XXX,

                        # Total runtime in hours.
                        "Runtime (Hours)": XXX,
                    },
                },
                ...
            ],
            ...
        },
        ...
    }

## Generating figures

Run these commands to generate all figures associated with the paper (replace
`figure_data.json` with the path to your figure_data). The default values are
set such that these commands generate the versions shown in the paper. Run all
Python commands in the Singularity container associated with this project, e.g.
run `make shell` to start a shell in the container and run the commands in that
shell.

    # For the comparison figure.
    python -m src.analysis.figures comparison figure_data.json

    # For the higher-res version of the comparison figure.
    python -m src.analysis.figures comparison_high_res figure_data.json

    # To generate the latex source for the tables in the paper.
    python -m src.analysis.figures table figure_data.json

    # To generate statistical test results.
    python -m src.analysis.figures tests figure_data.json

If including the Latex files output by these commands, make sure to also put
these commands in your paper:

    \usepackage{booktabs}
    \usepackage{multirow}
    \usepackage{array}
    \newcolumntype{L}[1]
        {>{\raggedright\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
    \newcolumntype{C}[1]
        {>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
    \newcolumntype{R}[1]
        {>{\raggedleft\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}

## Help

Run the following for more help:

    python -m src.analysis.figures COMMAND --help
"""
import itertools
import json
import pickle as pkl
import shutil
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable, List

import fire
import gin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin
import scipy.stats
import slugify
from logdir import LogDir
from loguru import logger
from ruamel.yaml import YAML
from statsmodels.graphics.factorplots import interaction_plot

import seaborn as sns
from src.analysis.utils import (is_me_es, load_experiment, load_me_es_objs,
                                load_metrics)
from src.mpl_styles import QUALITATIVE_COLORS
from src.mpl_styles.utils import mpl_style_file

# Metrics which we record in figure_data.json but do not plot.
METRIC_BLACKLIST = [
    "Normalized QD Score",
    "Normalized Mean Elite Robustness",
    "Normalized QD Score AUC",
]

# Reordered version of Seaborn colorblind palette.
COLORBLIND_REORDERED = np.array(
    sns.color_palette("colorblind"))[[0, 1, 2, 8, 4, 3, 5, 6, 7, 9]]


def load_manifest(manifest: str, entry: str = "Paper"):
    """Loads the data and root directory of the manifest."""
    manifest = Path(manifest)
    data = YAML().load(manifest)[entry]
    root_dir = manifest.parent
    return data, root_dir


def exclude_from_manifest(paper_data, env, algo):
    """Whether to exclude the given algo."""
    return "exclude" in paper_data[env]["algorithms"][algo]


def verify_manifest(paper_data, root_dir: Path, reps: int):
    """Checks logging directories for correctness."""
    for env in paper_data:
        for algo in paper_data[env]["algorithms"]:
            if exclude_from_manifest(paper_data, env, algo):
                continue

            results = paper_data[env]["algorithms"][algo]
            name = f"(Env: {env} Algo: {algo})"

            if reps is not None:
                # Check that the reps are correct.
                assert len(results) == reps, \
                    f"{name} {len(results)} dirs, needs {reps}"

            # Check logging dirs are unique.
            logdirs = [d["dir"] for d in results]
            assert len(logdirs) == len(set(logdirs)), \
                f"{name} {len(logdirs)} dirs listed, {len(set(logdirs))} unique"

            # Check seeds are unique.
            seeds = [d["seed"] for d in results]
            assert len(seeds) == len(set(seeds)), \
                f"{name} {len(seeds)} seeds listed, {len(set(seeds))} unique"

            # Check seeds match.
            for d in results:
                logdir = LogDir("tmp", custom_dir=root_dir / d["dir"])
                actual_seed = int(logdir.pfile("seed").open("r").read())
                assert actual_seed == d["seed"], \
                    (f"{name} Seed for {logdir} ({d['seed']}) does not match "
                     f"actual seed ({actual_seed})")


def find_min(manifest: str):
    """Finds the min obj that was inserted into the archive in any experiment.

    This minimum objective is calculated for each of the environments. It is
    then used e.g. in computing the QD score.

    Args:
        manifest: Path to YAML file holding paths to logging directories.
    """
    paper_data, root_dir = load_manifest(manifest)
    data = defaultdict(list)
    for env in paper_data:
        for algo in paper_data[env]["algorithms"]:
            if exclude_from_manifest(paper_data, env, algo):
                continue
            data[env].extend(
                [d["dir"] for d in paper_data[env]["algorithms"][algo]])

    for env in data:
        logdirs = data[env]
        min_score = np.inf
        min_logdir = None

        for d in logdirs:
            logdir = load_experiment(root_dir / d)

            if is_me_es():
                archive_history = load_me_es_objs(
                    logdir, return_archive=True).history()
            else:
                archive_type = str(gin.query_parameter("Manager.archive_type"))
                if archive_type != "@GridArchive":
                    print(logdir.logdir)
                    raise TypeError(f"Unknown archive type {archive_type}")

                archive_history_path = logdir.pfile("archive_history.pkl")
                with archive_history_path.open("rb") as file:
                    archive_history = pkl.load(file)

            for gen_history in archive_history:
                for obj, _ in gen_history:
                    min_score = min(min_score, obj)
                    min_logdir = root_dir / d

        logger.info("{}: {} ({})", env, min_score, min_logdir)


def collect(manifest: str, reps: int = 5, output: str = "figure_data.json"):
    """Collects data from logging directories and aggregates into a single JSON.

    Args:
        manifest: Path to YAML file holding paths to logging directories.
        reps: Number of times each experiment should be repeated.
        output: Path to save results.
    """
    logger.info("Loading manifest")
    paper_data, root_dir = load_manifest(manifest)

    logger.info("Verifying logdirs")
    verify_manifest(paper_data, root_dir, reps)

    # Mapping from the name in the raw metrics to the name in the output.
    metric_names = {
        "Actual QD Score": "QD Score",
        # "Normalized QD Score" gets added when we calculate "QD Score".
        "Archive Coverage": "Archive Coverage",
        "Best Performance": "Best Performance",
    }

    figure_data = {}

    logger.info("Loading Plot Data")
    for env in paper_data:
        figure_data[env] = OrderedDict()
        env_data = paper_data[env]

        for algo in env_data["algorithms"]:
            if exclude_from_manifest(paper_data, env, algo):
                continue

            figure_data[env][algo] = []

            for entry in env_data["algorithms"][algo]:
                figure_data[env][algo].append(cur_data := {})
                logdir = load_experiment(root_dir / entry["dir"])
                metrics = load_metrics(logdir)

                total_evals = metrics.get_single("Total Evals")["y"]
                cur_data["Evaluations"] = total_evals

                cur_data["series_metrics"] = {}
                qd_score_auc, norm_qd_score_auc = None, None
                for actual_name, figure_name in metric_names.items():
                    data = metrics.get_single(actual_name)

                    if actual_name == "Actual QD Score":
                        # Adjust QD score with new min -- we exclude the first
                        # element since it is always 0.
                        archive_size = np.array(
                            metrics.get_single("Archive Size")["y"])[1:]
                        qd_score = np.array(data["y"])[1:]
                        qd_score = (
                            qd_score + archive_size *
                            (env_data["old_min_obj"] - env_data["min_obj"]))
                        data["y"][1:] = qd_score.tolist()
                        cur_data["series_metrics"][figure_name] = data["y"]

                        # Also add in normalized QD score.
                        max_qd_score = env_data["archive_size"] * (
                            env_data["max_obj"] - env_data["min_obj"])
                        norm_qd_score = (np.array(data["y"]) /
                                         max_qd_score).tolist()
                        cur_data["series_metrics"][
                            "Normalized QD Score"] = norm_qd_score

                        # Finally, add in AUCs - for this paper, it's reasonable
                        # to assume that every generation has the same number of
                        # evals.
                        evals_per_gen = total_evals[-1] / (len(total_evals) - 1)
                        qd_score_auc = sum(qd_score) * evals_per_gen
                        norm_qd_score_auc = sum(norm_qd_score) * evals_per_gen
                    else:
                        cur_data["series_metrics"][figure_name] = data["y"]

                # Load robustness from the data created by robustness.py.
                robustness_file = logdir.file(
                    f"archive/archive_{len(total_evals) - 1}_robustness.pkl")
                robustness = pd.read_pickle(
                    robustness_file)["robustness"].mean()

                cur_data["point_metrics"] = {
                    "Mean Elite Robustness":
                        robustness,
                    "Normalized Mean Elite Robustness":
                        robustness /
                        (env_data["max_obj"] - env_data["min_obj"]),
                    "Runtime (Hours)":
                        metrics.get_single("Cum Time")["y"][-1] / 3600,
                    "QD Score AUC":
                        qd_score_auc,
                    "Normalized QD Score AUC":
                        norm_qd_score_auc,
                }

    logger.info("Saving to {}", output)
    with open(output, "w") as file:
        json.dump(figure_data, file)
    logger.info("Done")


def legend_info(names: Iterable, palette: dict, markers: dict):
    """Creates legend handles and labels for the given palette and markers.

    Yes, this is kind of a hack.
    """
    _, ax = plt.subplots(1, 1)
    for name in names:
        # We just need the legend handles, so the plot itself is bogus.
        ax.plot(
            [0],
            [0],
            label=name,
            color=palette[name],
            marker=markers[name],
            markeredgewidth="0.75",
            markeredgecolor="white",
        )
    return ax.get_legend_handles_labels()


def load_figure_data(figure_data: str):
    with open(figure_data, "r") as file:
        return json.load(file)


def metric_from_entry(entry, metric):
    """Retrieves the metric from either series_metrics or point_metrics.

    entry is a dict in the list associated with figure_data[env][algo]
    """
    return (entry["series_metrics"][metric][-1] if metric
            in entry["series_metrics"] else entry["point_metrics"][metric])


def comparison(
    figure_data: str = "./figure_data.json",
    outputs: List[str] = ("comparison.pdf", "comparison.png"),
    palette: str = "colorblind_reordered",
    height: float = 1.9,
    plot_every: int = 25,
):
    """Plots the figure comparing metrics of all algorithms.

    Args:
        figure_data: Path to JSON file with figure data.
        outputs: File to output the figure to.
        palette: Either a Seaborn color palette, "qualitative_colors" for
            QUALITATIVE_COLORS, or "colorblind_reordered" for
            COLORBLIND_REORDERED.
        height: Height in inches of each plot.
        plot_every: How frequently to plot points, e.g. plot every 100th point.
    """
    figure_data = load_figure_data(figure_data)

    plot_data = {
        "Environment": [],
        "Algorithm": [],
        "Metric": [],
        "Evaluations": [],
        "Score": [],
    }

    logger.info("Loading Plot Data")
    all_algos = OrderedDict()  # Set of all algorithms, ordered by listing.
    for env in figure_data:
        for algo in figure_data[env]:
            all_algos[algo] = None
            for entry in figure_data[env][algo]:
                # Has a length of generations + 1, since we add an extra 0 at
                # the start.
                evals = np.asarray(entry["Evaluations"])

                entry_metrics = entry["series_metrics"]

                for metric in entry_metrics:
                    if metric in METRIC_BLACKLIST:
                        continue

                    # Plot fewer data points to reduce file size.

                    # Metrics may have length of generations or generations + 1,
                    # as only some metrics (like archive size) add a 0 at the
                    # start.
                    metric_data = entry_metrics[metric]
                    raw_len = len(metric_data)
                    not_use_zero = int(len(metric_data) != len(evals))
                    gens = len(evals) - 1
                    # Start at 0 or 1 and end at gens.
                    x_data = np.arange(not_use_zero, gens + 1)

                    idx = list(range(0, raw_len, plot_every))
                    if idx[-1] != raw_len - 1:
                        # Make sure last index is included.
                        idx.append(raw_len - 1)

                    indexed_x_data = x_data[idx]
                    indexed_evals = evals[indexed_x_data]
                    indexed_metric_data = np.asarray(entry_metrics[metric])[idx]
                    data_len = len(indexed_evals)

                    plot_data["Environment"].append(np.full(data_len, env))
                    plot_data["Algorithm"].append(np.full(data_len, algo))
                    plot_data["Metric"].append(np.full(data_len, metric))
                    plot_data["Evaluations"].append(indexed_evals)
                    plot_data["Score"].append(indexed_metric_data)

    # Flatten everything so that Seaborn understands it.
    for d in plot_data:
        plot_data[d] = np.concatenate(plot_data[d])

    logger.info("Generating Plot")
    with mpl_style_file("simple.mplstyle") as f:
        with plt.style.context(f):
            if palette == "qualitative_colors":
                colors = QUALITATIVE_COLORS
            elif palette == "colorblind_reordered":
                # Rearrange the color-blind template.
                colors = COLORBLIND_REORDERED
            else:
                colors = sns.color_palette(palette)

            palette = dict(zip(all_algos, colors))
            markers = dict(zip(all_algos, itertools.cycle("oD^vXP")))

            # This takes a while since it has to generate the bootstrapped
            # confidence intervals.
            grid = sns.relplot(
                data=plot_data,
                x="Evaluations",
                y="Score",
                hue="Algorithm",
                style="Algorithm",
                row="Metric",
                col="Environment",
                kind="line",
                # TODO: Remove this note when Seaborn 0.12.0 is released.
                # errorbar is a feature in the development version of Seaborn.
                errorbar="se",
                markers=markers,
                markevery=(0.5, 10.0),
                dashes=False,
                height=height,
                aspect=1.61803,  # Golden ratio.
                facet_kws={"sharey": False},
                palette=palette,
                legend=False,
            )

            # Set titles to be the env name.
            grid.set_titles("{col_name}")

            # Turn off titles below top row (no need to repeat).
            for ax in grid.axes[1:].ravel():
                ax.set_title("")

            # Set the labels along the left column to be the name of the metric.
            left_col = list(figure_data)[0]
            for (row_val, col_val), ax in grid.axes_dict.items():
                ax.set_axisbelow(True)
                if col_val == left_col:
                    ax.set_ylabel(row_val, labelpad=10.0)

            # Add legend and resize figure to fit it.
            grid.fig.legend(*legend_info(all_algos, palette, markers),
                            bbox_to_anchor=[0.5, 1.0],
                            loc="upper center",
                            ncol=(len(palette) + 1) // 2)
            fig_width, fig_height = grid.fig.get_size_inches()
            legend_height = 0.55
            grid.fig.set_size_inches(fig_width, fig_height + legend_height)

            # Save the figure.
            grid.fig.tight_layout(rect=(0, 0, 1, fig_height /
                                        (fig_height + legend_height)))
            for output in outputs:
                grid.fig.savefig(output, dpi=300)

    logger.info("Done")


def comparison_high_res(
    figure_data: str = "./figure_data.json",
    outputs: List[str] = ("comparison_high_res.pdf", "comparison_high_res.png"),
):
    """Generates the larger version of the figure for the supplemental material.

    Simply calls comparison with the appropriate args.
    """
    return comparison(figure_data, outputs, height=4, plot_every=5)


# Header lines for table files.
TABLE_HEADER = r"""
% THIS FILE IS AUTO-GENERATED. DO NOT MODIFY THIS FILE DIRECTLY.

"""


def table(figure_data: str = "figure_data.json",
          transpose: bool = True,
          output: str = "results_table.tex"):
    """Creates Latex tables showing final values of metrics.

    Make sure to include the "booktabs" and "array" package in your Latex
    document.

    With transpose=False, a table is generated for each environment. Each table
    has the algorithms as rows and the metrics as columns.

    With transpose=True, a table is generated for each metric. Each table has
    the algorithms as rows and the environments as columns.

    Args:
        figure_data: Path to JSON file with figure data.
        transpose: See above.
        output: Path to save Latex table.
    """
    figure_data = load_figure_data(figure_data)

    # Safe to assume all envs have same metrics.
    first_env = list(figure_data)[0]
    first_algo = list(figure_data[first_env])[0]
    first_entry = figure_data[first_env][first_algo][0]
    metric_names = (list(first_entry["series_metrics"]) +
                    list(first_entry["point_metrics"]))
    for name in METRIC_BLACKLIST:
        metric_names.remove(name)
    if "QD Score AUC" in metric_names:
        # Move QD Score AUC immediately after QD Score.
        metric_names.remove("QD Score AUC")
        metric_names.insert(metric_names.index("QD Score") + 1, "QD Score AUC")
    logger.info("Metric names: {}", metric_names)

    table_data = {}
    logger.info("Gathering table data")
    for env in figure_data:
        table_data[env] = pd.DataFrame(index=list(figure_data[env]),
                                       columns=metric_names,
                                       dtype=str)
        for algo in figure_data[env]:
            for metric in metric_names:
                if metric in first_entry["series_metrics"]:
                    final_metric_vals = np.array([
                        entry["series_metrics"][metric][-1]
                        for entry in figure_data[env][algo]
                    ])
                else:
                    final_metric_vals = np.array([
                        entry["point_metrics"][metric]
                        for entry in figure_data[env][algo]
                    ])

                if metric == "QD Score AUC":
                    # Special case because these values are so large.
                    table_data[env][metric][algo] = (
                        f"{final_metric_vals.mean() / 1e12:,.2f}")
                else:
                    table_data[env][metric][algo] = (
                        f"{final_metric_vals.mean():,.2f}")

    if transpose:
        # "Invert" table_data.
        table_data = {
            metric:
            pd.DataFrame({env: df[metric] for env, df in table_data.items()})
            for metric in metric_names
        }

    logger.info("Writing to {}", output)
    with open(output, "w") as file:
        file.write(TABLE_HEADER)
        for name, df in table_data.items():
            if name == "QD Score AUC":
                caption = name + " (multiple of $10^{12}$)"
            else:
                caption = name

            file.write("\\begin{table*}[t]\n")
            file.write("\\caption{" + caption + "}\n")
            file.write("\\label{table:" + slugify.slugify(name) + "}\n")
            file.write("\\begin{center}\n")
            file.write(
                df.to_latex(
                    column_format="l" + " R{0.9in}" * len(df.columns),
                    escape=False,
                ))
            file.write("\\end{center}\n")
            file.write("\\end{table*}\n")
            file.write("\n")

    logger.info("Done")


def calc_simple_main_effects(figure_data, anova_res, metric):
    """Calculates simple main effects in each environment.

    Reference:
    http://www.cee.uma.pt/ron/Discovering%20Statistics%20Using%20SPSS,%20Second%20Edition%20CD-ROM/Calculating%20Simple%20Effects.pdf
    """
    df_residual = anova_res["DF"][3]
    ms_residual = anova_res["MS"][3]

    data = {
        "Environment": ["Residual"],
        "SS": [anova_res["SS"][3]],
        "DF": [df_residual],
        "MS": [ms_residual],
        "F": [np.nan],
        "p-unc": [np.nan],
        "significant": [False],
    }

    for env in figure_data:
        data["Environment"].append(env)

        algos, metric_vals = [], []
        for algo in figure_data[env]:
            entry_metrics = [
                metric_from_entry(entry, metric)
                for entry in figure_data[env][algo]
            ]

            algos.extend([algo] * len(entry_metrics))
            metric_vals.extend(entry_metrics)

        one_way = pingouin.anova(
            dv=metric,
            between=["Algorithm"],
            data=pd.DataFrame({
                "Algorithm": algos,
                metric: metric_vals,
            }),
            detailed=True,
        )

        f_val = one_way["MS"][0] / ms_residual
        p_unc = scipy.stats.f(one_way["DF"][0], df_residual).sf(f_val)
        sig = p_unc < 0.05

        data["SS"].append(one_way["SS"][0])
        data["DF"].append(one_way["DF"][0])
        data["MS"].append(one_way["MS"][0])
        data["F"].append(f_val)
        data["p-unc"].append(p_unc)
        data["significant"].append(sig)

    return pd.DataFrame(data)


def run_pairwise_ttests(figure_data, metric):
    """Runs pairwise t-tests for the hypotheses in the paper."""
    metric_vals = {
        env: {
            algo: [
                metric_from_entry(entry, metric)
                for entry in figure_data[env][algo]
            ] for algo in figure_data[env]
        } for env in figure_data
    }

    def compare_to(main_algo, other_algos):
        results = {}
        for env in figure_data:
            df = pd.concat(
                [
                    pingouin.ttest(
                        metric_vals[env][main_algo],
                        metric_vals[env][algo],
                        paired=False,
                        alternative="two-sided",
                    )[["T", "dof", "alternative", "p-val"]]
                    for algo in other_algos
                ],
                ignore_index=True,
            )
            df["significant"], df["p-val"] = pingouin.multicomp(
                list(df["p-val"]),
                alpha=0.05,
                method="bonf",
            )
            df = pd.concat(
                [
                    pd.DataFrame({
                        "Algorithm 1": [main_algo] * len(other_algos),
                        "Algorithm 2": other_algos,
                    }),
                    df,
                ],
                axis=1,
            )
            results[env] = df
        return results

    baselines = ["PGA-MAP-Elites", "ME-ES", "MAP-Elites"]
    if metric == "Normalized QD Score":
        return {
            "H1 - Comparing QD score between CMA-MEGA (ES) and baselines":
                compare_to("CMA-MEGA (ES)", baselines),
            "H2 - Comparing QD score between CMA-MEGA (TD3, ES) and baselines":
                compare_to("CMA-MEGA (TD3, ES)", baselines),
            "H3 - Comparing QD score between CMA-MEGA (ES) and "
            "CMA-MEGA (TD3, ES)":
                compare_to("CMA-MEGA (ES)", ["CMA-MEGA (TD3, ES)"]),
        }
    elif metric == "Normalized Mean Elite Robustness":
        return {
            ("Comparing mean elite robustness between MAP-Elites and CMA-MEGA "
             "variants"):
                compare_to("MAP-Elites",
                           ["CMA-MEGA (ES)", "CMA-MEGA (TD3, ES)"]),
        }
    elif metric == "Normalized QD Score AUC":
        return {
            "Comparing QD score AUC between PGA-ME and CMA-MEGA variants":
                compare_to("PGA-MAP-Elites",
                           ["CMA-MEGA (ES)", "CMA-MEGA (TD3, ES)"]),
        }
    else:
        raise NotImplementedError(f"No hypotheses for {metric}")


def tests_for_metric(figure_data, root_dir: Path, metric: str):
    """Saves tests for the metric into a subdirectory of root_dir.

    Returns results from various tests.
    """
    output = root_dir / slugify.slugify(metric)
    output.mkdir()

    data = {
        "Environment": [],
        "Algorithm": [],
        metric: [],
    }
    grouped_qd_scores = []

    logger.info("Loading {}", metric)
    for env in figure_data:
        for algo in figure_data[env]:
            entry_qd_scores = [
                metric_from_entry(entry, metric)
                for entry in figure_data[env][algo]
            ]

            data["Environment"].extend([env] * len(entry_qd_scores))
            data["Algorithm"].extend([algo] * len(entry_qd_scores))
            data[metric].extend(entry_qd_scores)

            grouped_qd_scores.append(entry_qd_scores)
    data = pd.DataFrame(data)

    logger.info("Drawing interaction plots")
    interaction_plot(
        data["Environment"],
        data["Algorithm"],
        data[metric],
        colors=COLORBLIND_REORDERED[:len(set(data["Algorithm"]))],
    )
    plt.savefig(output / "interaction_plot_1.png")
    interaction_plot(
        data["Algorithm"],
        data["Environment"],
        data[metric],
        colors=COLORBLIND_REORDERED[:len(set(data["Environment"]))],
    )
    plt.savefig(output / "interaction_plot_2.png")

    logger.info("Running ANOVA")
    anova_res = pingouin.anova(
        dv=metric,
        between=["Environment", "Algorithm"],
        data=data,
    )
    anova_res["significant"] = anova_res["p-unc"] < 0.05

    logger.info("Running simple main effects")
    simple_main = calc_simple_main_effects(figure_data, anova_res, metric)

    logger.info("Running pairwise t-tests")
    ttests = run_pairwise_ttests(figure_data, metric)
    ttest_str_parts = {
        hypothesis: '\n\n'.join(f"""\
#### {env}

{d.to_markdown()}
""" for env, d in results.items()) for hypothesis, results in ttests.items()
    }
    ttests_str = '\n\n'.join(f"""\
### {hypothesis}

{str_part}
""" for hypothesis, str_part in ttest_str_parts.items())

    logger.info("Checking homoscedasticity (equal variances)")
    var_test = pingouin.homoscedasticity(
        grouped_qd_scores,
        method="levene",
        alpha=0.05,
    )

    markdown_file = output / "README.md"
    logger.info("Writing results to {}", markdown_file)
    with open(markdown_file, "w") as file:
        file.write(f"""\
# Statistical Tests for {metric}

Significance tests done by checking if `p < 0.05`

## ANOVA

{anova_res.to_markdown()}

See [here](https://pingouin-stats.org/generated/pingouin.anova.html) for more
info on the pingouin ANOVA method.

**Table Columns:**

- `Source`: Factor names
- `SS`: Sums of squares
- `DF`: Degrees of freedom
- `MS`: Mean squares
- `F`: F-values
- `p-unc`: uncorrected p-values
- `np2`: Partial eta-square effect sizes

### Analyses

- Environment: $F({anova_res['DF'][0]}, {anova_res['DF'][3]}) = {anova_res['F'][0]:.2f}$
- Algorithm: $F({anova_res['DF'][1]}, {anova_res['DF'][3]}) = {anova_res['F'][1]:.2f}$
- Environment * Algorithm: $F({anova_res['DF'][2]}, {anova_res['DF'][3]}) = {anova_res['F'][2]:.2f}$

## Homoscedasticity (Equal Variances)

{var_test.to_markdown()}

## Interaction Plots

![Interaction Plot 1](./interaction_plot_1.png)

![Interaction Plot 2](./interaction_plot_2.png)

## Simple Main Effects

{simple_main.to_markdown()}

## Pairwise t-tests

(p-values Bonferroni corrected within each environment / simple main effect;
alpha is still 0.05)

{ttests_str}
""")

    write_tests_as_latex(metric, anova_res, simple_main, ttests, output)

    logger.info("Done")


def format_p_val(p, include_p=False, math=False):
    """Formats p-values for Latex."""
    if p < 0.001:
        p_str = f"{'p 'if include_p else '' }< 0.001"
    elif p >= 1.0:
        p_str = f"{'p = ' if include_p else ''}1"
    else:
        p_str = f"{'p = ' if include_p else ''}{p:.3f}"

    # Put significant p-values in bold.
    if p < 0.05:
        p_str = ("\\mathbf{" if math else "\\textbf{") + p_str + "}"

    return p_str


def write_tests_as_latex(metric, anova_res, simple_main, ttests, output: Path):
    """Write the tests to latex files in the output directory."""
    logger.info("Writing tests as latex")

    # Write ANOVA results as list.
    with open(output / "anova.tex", "w") as file:
        caption = f"Simple main effects for {CAPTION_NAME[metric]}"
        label = f"{slugify.slugify(metric)}-main"
        file.write(TABLE_HEADER)
        file.write("\\begin{itemize}\n")
        file.write("\\item Interaction effect: "
                   f"$F({anova_res['DF'][2]}, {anova_res['DF'][3]}) = "
                   f"{anova_res['F'][2]:.2f}, "
                   f"{format_p_val(anova_res['p-unc'][2], True, True)}$\n")
        file.write("\\item Simple main effects:\n")
        file.write("  \\begin{itemize}\n")

        err_deg = simple_main["DF"][0]
        only_main = simple_main.loc[1:]
        for env, df, f, p in zip(only_main["Environment"], only_main["DF"],
                                 only_main["F"], only_main["p-unc"]):
            file.write(
                f"  \\item {env}: "
                f"$F({df}, {err_deg}) = {f:.2f}, {format_p_val(p, True, True)}$"
                "\n")

        file.write("  \\end{itemize}\n")
        file.write("\\end{itemize}\n")
        file.write("\n")

    # Write ttests as tables.
    table_i = 0
    for hypothesis, hyp_results in ttests.items():
        with open(output / f"ttests-{table_i}.tex", "w") as file:
            file.write(TABLE_HEADER)

            first_env_df = next(iter(hyp_results.values()))
            pval_df = pd.DataFrame(
                # Index is the algorithms.
                index=pd.MultiIndex.from_frame(
                    first_env_df[["Algorithm 1", "Algorithm 2"]]),
                # Columns are environments.
                columns=list(hyp_results),
                dtype=str,
            )

            for env, env_df in hyp_results.items():
                for algo1, algo2, p in zip(env_df["Algorithm 1"],
                                           env_df["Algorithm 2"],
                                           env_df["p-val"]):
                    pval_df[env][algo1, algo2] = format_p_val(p)

            caption = f"{hypothesis}"
            label = f"{slugify.slugify(metric)}-ttests-{table_i}"
            table_i += 1

            file.write("\\begin{table*}[t]\n")
            file.write("\\caption{" + caption + "}\n")
            file.write("\\label{table:" + slugify.slugify(label) + "}\n")
            file.write("\\begin{center}\n")
            file.write(
                pval_df.to_latex(
                    column_format=" L{1.2 in}" * 2 +
                    " R{0.9in}" * len(pval_df.columns),
                    escape=False,
                    multirow=True,
                ))
            file.write("\\end{center}\n")
            file.write("\\end{table*}\n")
            file.write("\n")


# Name in table captions.
CAPTION_NAME = {
    "Normalized QD Score": "QD score",
    "Normalized QD Score AUC": "QD score AUC",
    "Normalized Mean Elite Robustness": "mean elite robustness",
}


def tests(figure_data: str = "figure_data.json", output: str = "stats_tests"):
    """Outputs information about statistical tests.

    We use Normalized scores since the ANOVA needs values to have the same
    scale.

    Args:
        figure_data: Path to JSON file with figure data.
        output: Directory to save (extended) outputs.
    """
    logger.info("Creating logging directory")
    output = Path(output)
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir()

    logger.info("Loading figure data")
    figure_data = load_figure_data(figure_data)

    with mpl_style_file("simple.mplstyle") as f:
        with plt.style.context(f):
            for metric in [
                    "Normalized QD Score",
                    "Normalized QD Score AUC",
                    "Normalized Mean Elite Robustness",
            ]:
                logger.info("===== {} =====", metric)
                tests_for_metric(figure_data, output, metric)


if __name__ == "__main__":
    fire.Fire({
        "find_min": find_min,
        "collect": collect,
        "comparison": comparison,
        "comparison_high_res": comparison_high_res,
        "table": table,
        "tests": tests,
    })
