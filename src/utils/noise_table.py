"""Provides a table of Gaussian noise for use by all workers.

This code is originally based on the implementation from the OpenAI ES paper:
https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py#L50
"""
import logging
from dataclasses import dataclass
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NoiseVector:
    """An entry of length dim in the noise table starting at index i."""
    i: int = None
    dim: int = None
    #: Use this to indicate the noise should be mirrored (multiplied by -1).
    mirror: bool = False


class NoiseTable:
    """A large array of Gaussian noise shared between all workers.

    Usage:
        After instantiating, call sample_index() to retrieve a noise index, then
        call get() to retrieve the noise.

    Args:
        seed: Seed for the Gaussian values. Defaults to 42 (i.e. the answer to
            the ultimate question).
        size: Number of values to generate in the array.
    """

    def __init__(self, seed: int = 42, size: int = 250_000_000):
        logger.info("Sampling %d numbers with seed %d", size, seed)
        rng = np.random.default_rng(seed)
        self.noise = rng.standard_normal(size, dtype=np.float32)
        logger.info("Sampled %d bytes", self.noise.size * 4)

    def get(self, i: int, dim: int) -> np.ndarray:
        """Returns an array of dim Gaussian values starting at index i."""
        return self.noise[i:i + dim]

    def get_vec(self, vec: NoiseVector) -> np.ndarray:
        """Same as get() but takes in NoiseVector object."""
        noise = self.get(vec.i, vec.dim)
        return -noise if vec.mirror else noise

    def sample_index(self,
                     rng: np.random.Generator,
                     dim: int,
                     batch_size: int = None) -> Union[int, np.ndarray]:
        """Selects a random index / indices that can be passed into get().

        Pass in batch_size to get a batch of indices. Should be either None for
        a single value or int for an array of values. See Generator.integers
        https://numpy.org/devdocs/reference/random/generated/numpy.random.Generator.integers.html
        for more info.
        """
        return rng.integers(0,
                            len(self.noise) - dim + 1,
                            size=batch_size,
                            dtype=np.int32)

    def sample_index_vec(
            self,
            rng: np.random.Generator,
            dim: int,
            batch_size: int = None) -> Union[NoiseVector, List[NoiseVector]]:
        """Same as sample_index() but returns a NoiseVector or list of
        NoiseVector.

        The `mirror` attribute is set to False in all NoiseVector's output by
        this method.
        """
        if batch_size is None:
            return NoiseVector(self.sample_index(rng, dim, None), dim, False)
        if isinstance(batch_size, int):
            indices = self.sample_index(rng, dim, batch_size)
            return [NoiseVector(idx, dim, False) for idx in indices]

        raise RuntimeError(
            f"batch_size must be int or None but is {batch_size}")
