"""Tests for ObjectiveResult."""
import numpy as np
import pytest

from src.objectives.objective_result import ObjectiveResult


@pytest.mark.parametrize(
    "aggregation",
    ["mean", "median"],
)
def test_from_raw(aggregation):
    r = ObjectiveResult.from_raw(
        returns=np.array([1, 2, 3, 4], dtype=float),
        bcs=np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=float),
        lengths=np.array([10, 20, 30, 40], dtype=float),
        final_xpos=np.array([100, 200, 300, 400], dtype=float),
        obs_sum=np.ones((10, 10)),
        obs_sumsq=np.ones((20, 20)),
        obs_count=1000,
        opts={
            "aggregation": aggregation,
        },
    )

    mean = np.mean([1, 2, 3, 4])
    std = np.std([1, 2, 3, 4])

    assert np.all(np.isclose(r.returns, [1, 2, 3, 4]))
    assert np.all(np.isclose(r.bcs, [[1, 1], [2, 2], [3, 3], [4, 4]]))
    assert np.all(np.isclose(r.lengths, [10, 20, 30, 40]))
    assert np.all(np.isclose(r.final_xpos, [100, 200, 300, 400]))

    if aggregation == "mean":
        assert np.isclose(r.agg_return, mean)
        assert np.all(np.isclose(r.agg_bc, [mean, mean]))
        assert np.isclose(r.agg_length, 10 * mean)
        assert np.isclose(r.agg_final_xpos, 100 * mean)
    elif aggregation == "median":
        assert np.isclose(r.agg_return, 2.5)
        assert np.all(np.isclose(r.agg_bc, [2.5, 2.5]))
        assert np.isclose(r.agg_length, 25)
        assert np.isclose(r.agg_final_xpos, 250)

    assert np.isclose(r.std_return, std)
    assert np.all(np.isclose(r.std_bc, [std, std]))
    assert np.isclose(r.std_length, 10 * std)
    assert np.isclose(r.std_final_xpos, 100 * std)

    assert np.all(np.isclose(r.obs_sum, np.ones((10, 10))))
    assert np.all(np.isclose(r.obs_sumsq, np.ones((20, 20))))
    assert r.obs_count == 1000


def test_median_odd():
    """test_from_raw only tests case with even number of elements."""
    r = ObjectiveResult.from_raw(
        returns=np.array([1, 2, 3, 4, 5], dtype=float),
        bcs=np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], dtype=float),
        opts={"aggregation": "median"},
    )

    median = 3
    std = np.std([1, 2, 3, 4, 5])

    assert np.all(np.isclose(r.returns, [1, 2, 3, 4, 5]))
    assert np.all(np.isclose(r.bcs, [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]))

    assert np.isclose(r.agg_return, median)
    assert np.all(np.isclose(r.agg_bc, [median, median]))

    assert np.isclose(r.std_return, std)
    assert np.all(np.isclose(r.std_bc, [std, std]))
