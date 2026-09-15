# src/sqm_ai/build1/features.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CUTOFF_RULE = (
    "A feature on the row for month M may use only data "
    "timestamped strictly before the end of month M."
)


def rolling_slope(s: pd.Series, window: int = 12):
    """Least-squares slope of the last `window` points."""
    if window < 2:
        raise ValueError("window must be at least two months")
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denom = float((x * x).sum())

    def slope(v):
        v = np.asarray(v, dtype=float)
        if not np.isfinite(v).all():
            return np.nan
        return float((x * (v - v.mean())).sum() / denom)

    return s.rolling(window, min_periods=window).apply(
        slope, raw=True
    )


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Trend, structural and behavioural features."""
    df = df.sort_values(["supplier_id", "month"]).copy()
    df["month"] = pd.to_datetime(df["month"])
    if df.duplicated(["supplier_id", "month"]).any():
        raise ValueError("duplicate supplier-month row")
    ordinal = df.month.dt.year * 12 + df.month.dt.month
    gaps = ordinal.groupby(df.supplier_id).diff().dropna()
    if gaps.ne(1).any():
        raise ValueError(
            "reindex missing calendar months before rolling"
        )
    g = df.groupby("supplier_id", sort=False)

    df["ncr_per_1k_units"] = (
        1000
        * df["ncr_count_90d"]
        / (df["units_received_90d"].replace(0, np.nan))
    )
    df["pct_high_severity"] = df["sev3_count_90d"] / (
        df["ncr_count_90d"].replace(0, np.nan)
    )
    df["fpy_slope_12m"] = g["fpy"].transform(rolling_slope)
    df["audit_score_delta"] = df["audit_score_last"] - g[
        "audit_score_last"
    ].shift(1)
    df["month_end"] = df["month"] + pd.offsets.MonthBegin(1)
    df["years_as_supplier"] = (
        df["month_end"] - df["onboarded_at"]
    ).dt.days / 365.25
    total_spend = df.groupby("month")["spend_90d"].transform(
        "sum"
    )
    df["spend_share"] = df["spend_90d"] / total_spend.replace(
        0, np.nan
    )
    return df


def load_modelling_frame(df):
    """Select named predictors; reject immature or invalid targets."""
    from sqm_ai.features import CATEGORICAL, NUMERIC

    required = (
        NUMERIC
        + CATEGORICAL
        + [
            "sev3_next_90d",
            "harm_next_90d",
            "month",
            "label_end",
            "data_complete_through",
        ]
    )
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"missing modeling columns: {missing}")
    if df[["sev3_next_90d", "harm_next_90d"]].isna().any().any():
        raise ValueError(
            "remove right-censored rows before training"
        )
    if not df.label_end.le(df.data_complete_through).all():
        raise ValueError("forward label not yet complete")
    if not df.sev3_next_90d.isin([0, 1]).all():
        raise ValueError("classification target must be binary")
    if (
        not np.isfinite(
            df.harm_next_90d.to_numpy(dtype=float)
        ).all()
        or (df.harm_next_90d < 0).any()
    ):
        raise ValueError(
            "severity-sum target must be finite and nonnegative"
        )
    X = df[NUMERIC + CATEGORICAL].copy()
    if np.isinf(X[NUMERIC].to_numpy(dtype=float)).any():
        raise ValueError("infinite feature value")
    return (
        X,
        df.sev3_next_90d.astype(int),
        df.harm_next_90d.astype(float),
    )


def build_features(
    as_of,
    data_path="data/supplier_month_all.parquet",
    engine=None,
):
    """Read a versioned month-end feature snapshot for local scoring.

    This adapter does not reconstruct historical source records.
    The production extractor must enforce event AND availability times.
    """
    as_of = pd.Timestamp(as_of)
    if pd.isna(as_of) or as_of.tzinfo is not None:
        raise ValueError("use a valid naive UTC snapshot date")
    if engine is not None:
        return build_database_features(as_of, engine)
    history = pd.read_parquet(Path(data_path))
    history["month"] = pd.to_datetime(history.month)
    cutoff = history.month + pd.offsets.MonthBegin(1)
    eligible = history[cutoff <= as_of]
    if eligible.empty:
        raise ValueError(
            "no completed month available at scoring date"
        )
    last = eligible.month.max()
    expected = pd.Timestamp(as_of).to_period("M").start_time
    if last + pd.offsets.MonthBegin(1) != expected:
        raise ValueError(
            "expected completed-month snapshot is missing"
        )
    result = eligible[eligible.month.eq(last)].copy()
    if result.supplier_id.duplicated().any():
        raise ValueError("duplicate supplier at scoring cutoff")
    result["snapshot_cutoff"] = last + pd.offsets.MonthBegin(1)
    return result.sort_values("supplier_id").reset_index(
        drop=True
    )


def build_database_features(as_of, engine):
    """Build event-time features from an authorized as-of source copy."""
    from sqlalchemy import text

    sql = (
        Path(__file__).parent / "sql" / "features.sql"
    ).read_text()
    with engine.begin() as conn:
        for statement in sql.split(";"):
            if statement.strip():
                conn.execute(text(statement))
    history = pd.read_sql_query(
        "SELECT * FROM sqm.supplier_features ORDER BY supplier_id, month",
        engine,
        parse_dates=["month", "onboarded_at", "audit_date_last"],
    )
    history = add_derived(history)
    history["limited_history"] = (
        history.groupby("supplier_id").cumcount() < 5
    )
    cutoff = history.month + pd.offsets.MonthBegin(1)
    eligible = history[cutoff <= pd.Timestamp(as_of)]
    if eligible.empty:
        raise ValueError("no completed source month available")
    last = eligible.month.max()
    expected = pd.Timestamp(as_of).to_period("M").start_time
    if last + pd.offsets.MonthBegin(1) != expected:
        raise ValueError(
            "expected completed-month snapshot is missing"
        )
    result = eligible[eligible.month.eq(last)].copy()
    result["snapshot_cutoff"] = last + pd.offsets.MonthBegin(1)
    return result.sort_values("supplier_id").reset_index(
        drop=True
    )
