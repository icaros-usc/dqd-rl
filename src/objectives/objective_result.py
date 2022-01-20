"""Class representing the results of an objective function evaluation."""
from dataclasses import dataclass

import numpy as np


def maybe_mean(arr, indices=None):
    """Calculates mean of arr[indices] if possible.

    indices should be a list. If it is None, the mean of the whole arr is taken.
    """
    indices = (slice(len(arr))
               if arr is not None and indices is None else indices)
    return None if arr is None else np.mean(arr[indices], axis=0)


def maybe_median(arr, indices=None):
    """Same as maybe_mean but with median."""
    indices = (slice(len(arr))
               if arr is not None and indices is None else indices)
    return None if arr is None else np.median(arr[indices], axis=0)


def maybe_std(arr, indices=None):
    """Same as maybe_mean but with std."""
    indices = (slice(len(arr))
               if arr is not None and indices is None else indices)
    return None if arr is None else np.std(arr[indices], axis=0)


@dataclass
class ObjectiveResult:  # pylint: disable = too-many-instance-attributes
    """Represents `n` results from an objective function evaluation.

    `n` is typically the number of evals (n_evals).

    Different fields are filled based on the objective function.
    """

    ## Raw data ##

    # (n,) array with results of all evaluations.
    returns: np.ndarray = None
    # (n, behavior_dim) array with all the BCs.
    bcs: np.ndarray = None
    # (n,) array of trajectory lengths (measured in timesteps).
    lengths: np.ndarray = None
    # (n,) array of final x positions.
    final_xpos: np.ndarray = None

    ## Aggregate data ##

    agg_return: float = None
    agg_bc: np.ndarray = None  # (behavior_dim,) array
    agg_length: float = None
    agg_final_xpos: float = None

    ## Measures of spread ##

    std_return: float = None
    std_bc: np.ndarray = None  # (behavior_dim,) array
    std_length: float = None
    std_final_xpos: float = None

    ## Observation statistics - shape based on env obs space (if applicable) ##

    obs_sum: np.ndarray = None  # Sum of observations.
    obs_sumsq: np.ndarray = None  # Sum of squared observations.
    obs_count: int = None  # Number of observations included.

    ## RL experience ##

    # Tuples of experience for RL training (typically comes from GymControl).
    experience: list = None

    @staticmethod
    def from_raw(
        returns,
        bcs=None,
        lengths=None,
        final_xpos=None,
        obs_sum=None,
        obs_sumsq=None,
        obs_count=None,
        experience=None,
        opts=None,
    ):
        """Constructs an ObjectiveResult from raw data.

        `opts` is a dict with several configuration options. It may be better as
        a gin parameter, but since ObjectiveResult is created on workers, gin
        parameters are unavailable (unless we start loading gin on workers too).
        Options in `opts` are:

            `aggregation` (default="mean"): How each piece of data should be
                aggregated into single values. Options are:
                - "mean": Take the mean, e.g. mean BC, mean length, etc.
                - "median": Take the median, e.g. median BC (element-wise),
                  median length, median return, etc.
        """
        # Handle config options.
        opts = {} if opts is None else opts
        opts.setdefault("aggregation", "mean")

        assert returns is not None, "returns cannot be None"
        n_evals = len(returns)
        std_indices = np.arange(n_evals)  # Indices for calculating std.

        # Calculate aggregate measures (agg_*).
        if opts["aggregation"] == "mean":
            agg_return = maybe_mean(returns)
            agg_bc = maybe_mean(bcs)
            agg_length = maybe_mean(lengths)
            agg_final_xpos = maybe_mean(final_xpos)
        elif opts["aggregation"] == "median":
            agg_return = maybe_median(returns)
            agg_bc = maybe_median(bcs)
            agg_length = maybe_median(lengths)
            agg_final_xpos = maybe_median(final_xpos)
        else:
            raise ValueError(f"Unknown aggregation {opts['aggregation']}")

        return ObjectiveResult(
            returns=returns,
            bcs=bcs,
            lengths=lengths,
            final_xpos=final_xpos,
            agg_return=agg_return,
            agg_bc=agg_bc,
            agg_length=agg_length,
            agg_final_xpos=agg_final_xpos,
            std_return=maybe_std(returns, std_indices),
            std_bc=maybe_std(bcs, std_indices),
            std_length=maybe_std(lengths, std_indices),
            std_final_xpos=maybe_std(final_xpos, std_indices),
            obs_sum=obs_sum,
            obs_sumsq=obs_sumsq,
            obs_count=obs_count,
            experience=experience,
        )
