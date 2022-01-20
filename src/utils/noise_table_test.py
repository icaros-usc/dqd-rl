"""Tests for NoiseTable."""
import numpy as np

from src.utils.noise_table import NoiseTable


def test_mirrored_sample():
    table = NoiseTable(size=1000)
    rng = np.random.default_rng()
    vec = table.sample_index_vec(rng, 100, None)
    noise = table.get_vec(vec)
    vec.mirror = True
    mirrored_noise = table.get_vec(vec)

    assert (noise == -mirrored_noise).all()
