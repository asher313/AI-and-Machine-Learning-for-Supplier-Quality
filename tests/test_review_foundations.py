"""Regression cases from the technical review, chapters 1-6."""
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from sqm_ai.anomalies import fpy_slope, pareto, rolling_z
from sqm_ai.features import zscore
from sqm_ai.ncr import NCRInput
from sqm_ai.stats import bootstrap_ci, bootstrap_diff_ci


def test_pareto_includes_crossing_supplier():
    frame = pd.DataFrame({"supplier_id": ["A", "B", "C"],
                          "ncr_count": [60, 25, 15]})
    assert pareto(frame) == ["A", "B"]
    frame["ncr_count"] = [80, 10, 10]
    assert pareto(frame) == ["A"]
    frame["ncr_count"] = 0
    assert pareto(frame) == []


def test_ncr_identifier_rejects_trailing_newline():
    row = dict(ncr_id="NCR-2026-0042", supplier_id="S-0417",
               defect_description="hole", quantity=1,
               discovered_at="2026-01-01T00:00:00")
    for field in ("ncr_id", "supplier_id"):
        bad = {**row, field: row[field] + "\n"}
        with pytest.raises(ValidationError):
            NCRInput(**bad)


def test_rolling_baseline_excludes_current_observation():
    values = np.tile([0.9, 0.95], 20).astype(float)
    frame = pd.DataFrame({"supplier_id": "A",
        "day": pd.date_range("2026-01-01", periods=40),
        "fpy": values})
    before = rolling_z(frame)
    frame.loc[39, "fpy"] = 0.1
    after = rolling_z(frame)
    assert after.loc[39, "roll_mean"] == before.loc[39, "roll_mean"]
    assert after.loc[39, "roll_std"] == before.loc[39, "roll_std"]
    assert after.loc[39, "z"] < -3


def test_rolling_uses_dates_and_flags_flat_baseline_change():
    frame = pd.DataFrame({"supplier_id": "A",
        "day": pd.date_range("2026-01-01", periods=31),
        "fpy": [1.0] * 30 + [0.7]})
    result = rolling_z(frame)
    assert result.iloc[-1]["flat_baseline_change"]
    frame.loc[30, "day"] = pd.Timestamp("2026-05-01")
    assert np.isnan(rolling_z(frame).iloc[-1]["z"])
    assert not rolling_z(frame).iloc[-1]["flat_baseline_change"]


def test_slope_preserves_missing_month_gaps():
    frame = pd.DataFrame({"month": pd.date_range(
        "2025-01-01", periods=6, freq="2MS"),
        "fpy": 0.99 - np.arange(6) * 0.02})
    assert fpy_slope(frame) == pytest.approx(-0.01)


@pytest.mark.parametrize("values", [[], [1], [np.nan, 1], [1, 1]])
def test_zscore_rejects_invalid_samples(values):
    with pytest.raises(ValueError):
        zscore(np.asarray(values))


def test_bootstrap_parameters_and_reproducibility():
    data = np.arange(10.0)
    assert bootstrap_ci(data, np.mean, n_boot=50) == \
        bootstrap_ci(data, np.mean, n_boot=50)
    with pytest.raises(ValueError):
        bootstrap_ci(data, np.mean, alpha=1)
    with pytest.raises(ValueError):
        bootstrap_diff_ci(data, data[:1], np.mean)
