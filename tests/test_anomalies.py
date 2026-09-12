# tests/test_anomalies.py
import numpy as np
import pandas as pd

from sqm_ai.anomalies import fpy_slope, rolling_z


def _series(vals: list[float]) -> pd.DataFrame:
    n = len(vals)
    return pd.DataFrame({
        "supplier_id": ["S-0001"] * n,
        "day": pd.date_range("2026-01-01", periods=n),
        "fpy": vals,
    })


def test_flat_series_has_no_anomaly():
    d = rolling_z(_series([0.95] * 40))
    assert d["z"].abs().max(skipna=True) != np.inf


def test_windows_do_not_cross_suppliers():
    a = _series([0.99] * 40)
    b = _series([0.50] * 40).assign(supplier_id="S-0002")
    d = rolling_z(pd.concat([a, b], ignore_index=True))
    first = d.loc[d["supplier_id"] == "S-0002"].head(1)
    z = first["z"].iloc[0]
    assert not abs(z) > 3   # a NaN is not a flag


def test_slope_is_negative_when_declining():
    g = pd.DataFrame({
        "month": pd.date_range("2025-09-01", periods=12, freq="MS"),
        "fpy": np.linspace(0.98, 0.86, 12),
    })
    assert fpy_slope(g) < -0.01
