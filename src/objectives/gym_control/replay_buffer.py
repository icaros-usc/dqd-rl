"""TD3 for GymControl policies."""
from collections import namedtuple

import numpy as np
import torch

Experience = namedtuple("Experience",
                        ["obs", "action", "reward", "bcs", "next_obs", "done"])

# Used for batches of items, e.g. a batch of obs, a batch of actions.
BatchExperience = namedtuple("BatchExperience", Experience._fields)


class ReplayBuffer:
    """Stores experience for training RL algorithms.

    Based on TD3 implementation and PGA-MAP-Elites implementation:
    https://github.com/sfujim/TD3/blob/master/utils.py
    https://github.com/ollenilsson19/PGA-MAP-Elites/blob/master/utils.py
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        action_shape: tuple,
        behavior_dim: int,
        seed: int = None,
    ):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.additions = 0

        self.obs = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.action = np.empty((capacity, *action_shape), dtype=np.float32)
        self.reward = np.empty(capacity, dtype=np.float32)
        self.bcs = np.empty((capacity, behavior_dim), dtype=np.float32)
        self.next_obs = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.done = np.empty(capacity, dtype=np.float32)

        self.rng = np.random.default_rng(seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def add(self, e: Experience):
        """Adds experience to the buffer."""
        self.obs[self.ptr] = e.obs
        self.action[self.ptr] = e.action
        self.next_obs[self.ptr] = e.next_obs
        self.reward[self.ptr] = e.reward
        self.bcs[self.ptr] = e.bcs
        self.done[self.ptr] = e.done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.additions += 1

    def __len__(self):
        """Number of Experience in the buffer."""
        return self.size

    def sample_tensors(self, n: int):
        """Same as sample() but returns tensors with each item in batch."""
        if len(self) == 0:
            raise ValueError("No entries currently in ReplayBuffer.")

        # This is used in the TD3 and PGA-ME implementation - if the buffer is
        # big enough, sampling duplicates is not an issue.
        # https://github.com/sfujim/TD3/blob/385b33ac7de4767bab17eb02ade4a268d3e4e24f/utils.py#L32
        # One concern with this is that the indices are out of range. However,
        # since we are only adding to the buffer, indices in the range [ 0,
        # len(self) ) will always be occupied. If we also remove from the
        # buffer, then we would want to have some offset here.
        indices = self.rng.integers(len(self), size=n)

        return BatchExperience(
            torch.as_tensor(self.obs[indices], device=self.device),
            torch.as_tensor(self.action[indices], device=self.device),
            torch.as_tensor(self.reward[indices], device=self.device),
            torch.as_tensor(self.bcs[indices], device=self.device),
            torch.as_tensor(self.next_obs[indices], device=self.device),
            torch.as_tensor(self.done[indices], device=self.device),
        )
