# src/sqm_ai/features.py
#
# Merged, in book order:
#   Chapter 4 "Asher at Northlake" — build_supplier_month()
#   Chapter 7.2                    — the column lists
#   Chapter 7.4                    — month_folds()
#   Chapter 7 "Asher at Northlake" — prep, the ColumnTransformer
# EXAMPLE_CANON: "Reusable pieces in src/sqm_ai/features.py:
# month_folds(...) and prep."
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sqlalchemy import create_engine, text

from sqm_ai.settings import get_settings

log = structlog.get_logger()
SQL_DIR = Path(__file__).parent / "sql"

# Chapter 7.2 — name the column types explicitly.
TARGET = "sev3_next_90d"
KEYS = ["supplier_id", "month"]
NUMERIC = [
    "ncr_count_90d", "ncr_per_1k_units", "avg_severity_90d",
    "pct_high_severity", "otd_pct", "avg_days_late", "fpy",
    "fpy_slope_12m", "audit_score_last", "audit_score_delta",
    "years_as_supplier", "spend_share", "car_response_days",
    "car_effectiveness",
]
CATEGORICAL = ["tier"]


def zscore(x: np.ndarray) -> np.ndarray:
    """Standardize a finite, nonconstant numeric vector."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("need two or more finite values")
    sd = x.std()
    if sd <= np.finfo(float).eps * max(1.0, abs(x.mean())):
        raise ValueError("constant column")
    return (x - x.mean()) / sd


def build_supplier_month(database_url: str,
                         start_month="2025-09-01",
                         end_month="2026-08-01") -> pd.DataFrame:
    """Atomically refresh rows while preserving dependent views."""
    start, end = pd.Timestamp(start_month), pd.Timestamp(end_month)
    if (start > end or start.day != 1 or end.day != 1
            or start != start.normalize() or end != end.normalize()
            or start.tzinfo is not None or end.tzinfo is not None):
        raise ValueError("use ordered, timezone-naive month-start dates")
    params = {"start_month": start.date(), "end_month": end.date()}
    engine = create_engine(database_url)
    sql = (SQL_DIR / "supplier_month.sql").read_text()
    try:
        with engine.begin() as conn:
            for stmt in sql.split(";"):
                if stmt.strip():
                    conn.execute(text(stmt), params)
            df = pd.read_sql_query(
                "SELECT * FROM sqm.supplier_month ORDER BY 1, 3",
                conn, parse_dates=["month"])
            if df.duplicated(["supplier_id", "month"]).any():
                raise ValueError("duplicate supplier-month grain")
    finally:
        engine.dispose()
    log.info("supplier_month_built", rows=len(df))
    return df


def month_folds(df: pd.DataFrame, n_splits: int = 5,
                test_size: int = 3, gap: int = 3):
    """Yield (train_idx, test_idx) over whole months.

    gap counts observed month groups, not exact 90-day periods.
    Validate label availability against prediction cutoffs too.
    """
    if df["month"].isna().any():
        raise ValueError("month cannot be missing")
    months = np.sort(df["month"].unique())
    tscv = TimeSeriesSplit(
        n_splits=n_splits, test_size=test_size, gap=gap)
    for tr_m, te_m in tscv.split(months):
        tr = np.flatnonzero(df["month"].isin(months[tr_m]))
        te = np.flatnonzero(df["month"].isin(months[te_m]))
        if tr.size == 0 or te.size == 0:
            raise ValueError("empty chronological fold")
        yield tr, te


# Chapter 7 "Asher at Northlake" — shared by Chapter 8's
# bake-off and Chapter 10's Build 1, so the preprocessing is
# shared rather than copied.
prep = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median",
                                 add_indicator=True)),
        ("scale", StandardScaler()),
    ]), NUMERIC),
    ("tier", OrdinalEncoder(categories=[["C", "B", "A"]],
                            handle_unknown="use_encoded_value",
                            unknown_value=-1), ["tier"]),
])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-month", default="2025-09-01")
    parser.add_argument("--end-month", default="2026-08-01")
    args = parser.parse_args()
    build_supplier_month(get_settings().database_url,
                         args.start_month, args.end_month)
