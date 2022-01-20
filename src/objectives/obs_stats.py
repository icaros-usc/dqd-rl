"""Provides utilities for observation normalization.

Observation normalization refers to normalizing observations to be z-scores by
subtracting the mean and dividing by standard deviation. Since the mean and
standard deviation are not known in advance, they have to be calculated online.
RunningObsStats keeps track of the global mean and standard deviation and is
intended to be put on the head node. Meanwhile, ObsStats is a smaller dataclass
intended to carry the stats to worker nodes. When a worker is done with
executions, it will return additional statistics via ObjectiveResult, and the
head node will add these to RunningObsStats with the increment() method.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ObsStats:
    """Small class with current observation stats."""
    mean: np.ndarray
    std: np.ndarray


class RunningObsStats:
    """Keeps track of stats about observations for observation normalization."""

    MIN_STD = 0.1

    def __init__(self, obs_shape: Tuple[int]):
        self._obs_shape = obs_shape

        # Sum of all observations.
        self._sum = np.zeros(obs_shape, dtype=np.float32)
        # Sum of square of all observations.
        self._sumsq = np.zeros(obs_shape, dtype=np.float32)
        # Number of observations included so far - floating point so that having
        # more observations is not (as much) of an issue.
        self._count = 0.0

        self._update()

    def _update(self):
        """Updates the mean and standard deviation.

        Standard deviation is guaranteed to be at least ObsStats.MIN_STD.
        """
        if self._count == 0:
            # With no info, the mean is assumed to be 0, and the standard
            # deviation is assumed to be 1.
            self._mean = self._sum
            self._std = np.ones(self._obs_shape, dtype=np.float32)
        else:
            self._mean = self._sum / self._count

            # https://en.wikipedia.org/wiki/Variance#Definition
            expected_sq = self._sumsq / self._count
            variance = expected_sq - np.square(self._mean)

            # Sometimes numerical stability results in the square of the mean
            # being slightly larger than expected_sq, making parts of the
            # variance negative.
            variance = np.maximum(variance, 0.0)

            # maximum preserves the shape of variance.
            self._std = np.maximum(np.sqrt(variance), self.MIN_STD)

    def increment(self, new_sum: np.ndarray, new_sumsq: np.ndarray,
                  new_count: int) -> Tuple[np.ndarray, np.ndarray]:
        """Increments the sum, sum of squared, and count by the values given.

        Returns:
            A tuple with:
            - The difference between the new mean and old mean
            - The difference between the new std and old std
        """
        self._sum += new_sum
        self._sumsq += new_sumsq
        self._count += new_count
        old_mean, old_std = self.mean.copy(), self.std.copy()
        self._update()
        return (self.mean - old_mean, self.std - old_std)

    @property
    def obs_shape(self) -> Tuple[int]:
        """Shape of the observation space."""
        return self._obs_shape

    @property
    def mean(self) -> np.ndarray:
        """The observation mean - dims are self.obs_shape."""
        return self._mean

    @property
    def std(self) -> np.ndarray:
        """The observation std - dims are self.obs_shape."""
        return self._std

    @property
    def count(self) -> int:
        """The number of observations counted so far."""
        return self._count

    def set_from_init(self, init_mean: np.ndarray, init_std: np.ndarray,
                      init_count: int):
        """Set the stats from the given info."""
        self._sum[:] = init_mean * init_count
        variance = np.square(init_mean) + np.square(init_std)
        self._sumsq[:] = variance * init_count
        self._count = init_count
        self._update()
