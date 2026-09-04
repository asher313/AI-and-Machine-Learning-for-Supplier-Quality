# tests/test_features.py
import numpy as np
import pytest

from sqm_ai.features import zscore


def test_zscore_has_zero_mean_unit_sd():
    x = np.array([0.95, 0.92, 0.88, 0.97, 0.90])
    z = zscore(x)
    assert z.mean() == pytest.approx(0.0, abs=1e-9)
    assert z.std() == pytest.approx(1.0, abs=1e-9)


def test_zscore_constant_column_raises():
    with pytest.raises(ValueError):
        zscore(np.array([0.9, 0.9, 0.9]))


def test_same_seed_same_sample():
    a = np.random.default_rng(42).normal(size=5)
    b = np.random.default_rng(42).normal(size=5)
    np.testing.assert_array_equal(a, b)
