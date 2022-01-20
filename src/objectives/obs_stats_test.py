"""Tests for RunningObsStats."""
import numpy as np
import pytest

from src.objectives.obs_stats import RunningObsStats

# pylint: disable = redefined-outer-name


@pytest.fixture(params=[(111,), (2, 2)])
def shape(request):
    """Shape of the observation space."""
    return request.param


def test_initial_stats(shape):
    stats = RunningObsStats(shape)
    assert stats.obs_shape == shape
    assert stats.mean.shape == shape
    assert stats.std.shape == shape
    assert (stats.mean == np.zeros(shape, dtype=np.float32)).all()
    assert (stats.std == np.ones(shape, dtype=np.float32)).all()


def test_add_one_stat(shape):
    stats = RunningObsStats(shape)
    obs = np.zeros(shape, dtype=np.float32)
    stats.increment(obs, obs**2, 1)

    assert np.isclose(stats.mean, obs).all()
    # std would be 0, but MIN_STD limits it
    assert np.isclose(stats.std, RunningObsStats.MIN_STD).all()


def test_add_more_stats(shape):
    stats = RunningObsStats(shape)
    obs1 = np.zeros(shape, dtype=np.float32)
    stats.increment(25 * obs1, 25 * (obs1**2), 25)
    obs2 = np.full(shape, 2.0, dtype=np.float32)
    mean_diff, std_diff = stats.increment(75 * obs2, 75 * (obs2**2), 75)

    # (75 * 2 + 25 * 0) / 100 = 1.5
    mean = np.full(shape, 1.5, dtype=np.float32)
    # sqrt((75 * 2**2 + 25 * 0**2) / 100 - 1.5**2)
    std = np.full(shape, np.sqrt(0.75), dtype=np.float32)

    assert np.isclose(mean_diff, mean).all()
    assert np.isclose(std_diff, std - RunningObsStats.MIN_STD).all()
    assert np.isclose(stats.mean, mean).all()
    assert np.isclose(stats.std, std).all()


def test_set_from_init(shape):
    stats = RunningObsStats(shape)
    mean = np.full(shape, 3.5, dtype=np.float32)
    std = np.full(shape, 1.0, dtype=np.float32)
    stats.set_from_init(mean, std, 100)

    assert np.isclose(stats.mean, mean).all()
    assert np.isclose(stats.std, std).all()
