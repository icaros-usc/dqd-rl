"""Provides ObjectiveBase."""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from src.objectives.objective_result import ObjectiveResult


class ObjectiveBase(ABC):
    """Base class for objective functions.

    Note that the seed is not passed into the constructor. The idea is that many
    different "users" can call this class, each with their own seed -- they will
    pass the seed into the method when they do so.

    Furthermore, each method should try to use settings from the config as much
    as possible, instead of relying in method parameters. This helps keep the
    API consistent.
    """

    @abstractmethod
    def __init__(self, config):
        """Initializes from a single config."""
        self._config = config

    @property
    def config(self):
        """The config for the module."""
        return self._config

    @abstractmethod
    def initial_solution(self, seed: Optional[int] = None) -> np.ndarray:
        """Returns an initial solution."""

    @abstractmethod
    def evaluate(self,
                 solution: np.ndarray,
                 n_evals: int,
                 seed: Optional[int] = None) -> ObjectiveResult:
        """Runs n_evals evaluations of the solution."""

    def jacobian_batch(self, batch_solutions: np.ndarray) -> np.ndarray:
        """Calculates the Jacobian for a batch of solutions.

        The API is weird because evaluate() takes in a single solution, while
        this method calculates gradients in batch. This is because while the
        evaluations are done on individual workers, the gradient is intended to
        be calculated on the master node (since the master is where we put the
        GPU).

        This method is not abstract because not all objectives will have a
        gradient available.

        Returns:
            Array of Jacobian matrices. Each matrix contains the Jacobian of the
            objective followed by the Jacobians of the measures.
        """
        raise NotImplementedError
