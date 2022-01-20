"""Tests for GymControl TD3."""
import torch

from src.objectives.gym_control.replay_buffer import (BatchExperience,
                                                      Experience, ReplayBuffer)

#
# ReplayBuffer
#


def test_inserts_at_most_max_elements():
    capacity = 5
    rb = ReplayBuffer(capacity, (), (), 1)
    for i in range(capacity + 1):
        rb.add(Experience(i, i, i, i, i, i))
    assert len(rb) == capacity


def test_sample_tensors():
    rb = ReplayBuffer(1, (), (), 1)
    for i in [1, 2]:
        rb.add(Experience(i, i, i, i, i, i))

    # Sampling 2 from a replay buffer of size 2 should give back both elements.
    x = torch.FloatTensor([2.0]).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    assert rb.sample_tensors(1) == BatchExperience(x, x, x, x, x, x)
