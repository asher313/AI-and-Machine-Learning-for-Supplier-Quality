# src/sqm_ai/build1/features.py
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

CUTOFF_RULE = (
    "A feature on the row for month M may use only data "
    "timestamped strictly before the end of month M."
)


def rolling_slope(s: pd.Series, window: int = 12):
    """Least-squares slope of the last `window` points."""
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = float((x * x).sum())

    def slope(v):
        v = np.asarray(v, dtype=float)
        if np.isnan(v).any():
            return np.nan
        return float((x * (v - v.mean())).sum() / denom)

    return s.rolling(window, min_periods=window).apply(
        slope, raw=True)


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Trend, structural and behavioural features."""
    df = df.sort_values(["supplier_id", "month"]).copy()
    g = df.groupby("supplier_id", sort=False)

    df["ncr_per_1k_units"] = 1000 * df["ncr_count_90d"] / (
        df["units_received_90d"].replace(0, np.nan))
    df["pct_high_severity"] = df["sev3_count_90d"] / (
        df["ncr_count_90d"].replace(0, np.nan))
    df["fpy_slope_12m"] = g["fpy"].transform(rolling_slope)
    df["audit_score_delta"] = (
        df["audit_score_last"]
        - g["audit_score_last"].shift(1))
    df["month_end"] = (
        df["month"] + pd.offsets.MonthBegin(1))
    df["years_as_supplier"] = (
        (df["month_end"] - df["onboarded_at"]).dt.days
        / 365.25)
    df["spend_share"] = df["spend_90d"] / df.groupby(
        "month")["spend_90d"].transform("sum")
    return df
