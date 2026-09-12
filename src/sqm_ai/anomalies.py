# Chapter 6 — Asher at Northlake
# TODO(book): condensed in Chapter 6 — complete before production use.
# src/sqm_ai/anomalies.py
from datetime import date

import numpy as np
import pandas as pd
import structlog
from scipy import stats
from sqlalchemy import create_engine

from sqm_ai.settings import get_settings

log = structlog.get_logger()
Z_THRESHOLD = 3.0
SLOPE_THRESHOLD = -0.01
MIN_DAYS = 200

DAILY_SQL = """
SELECT supplier_id,
       inspected_at::date AS day,
       AVG(passed::int)   AS fpy,
       COUNT(*)           AS units
  FROM sqm.inspections
 WHERE inspected_at >= DATE '2025-09-01'
 GROUP BY supplier_id, inspected_at::date
"""

MONTHLY_SQL = """
SELECT supplier_id, month, fpy, ncr_count
  FROM sqm.supplier_month
 ORDER BY supplier_id, month
"""


# Chapter 6 — Asher at Northlake (continued)
def rolling_z(
    daily: pd.DataFrame, window: int = 30
) -> pd.DataFrame:
    """Flag supplier-days whose FPY departs from trend."""
    if window < 20:
        raise ValueError("window must cover at least 20 days")
    d = daily.sort_values(["supplier_id", "day"]).copy()
    if d.duplicated(["supplier_id", "day"]).any():
        raise ValueError("one row per supplier and day required")
    d = d.set_index("day")
    g = d.groupby("supplier_id")["fpy"]
    d["roll_mean"] = g.transform(
        lambda s: s.rolling(
            f"{window}D", closed="left", min_periods=20
        ).mean()
    )
    d["roll_std"] = g.transform(
        lambda s: s.rolling(
            f"{window}D", closed="left", min_periods=20
        ).std()
    )
    d["z"] = (
        (d["fpy"] - d["roll_mean"])
        / d["roll_std"].replace(0, np.nan)
    )
    d["flat_baseline_change"] = (
        d["roll_std"].le(1e-12)
        & (d["fpy"] - d["roll_mean"]).abs().gt(1e-12)
    )
    return d.reset_index()


def fpy_slope(g: pd.DataFrame) -> float:
    """Slope of FPY per month; negative means degrading."""
    g = g.dropna(subset=["fpy", "month"]).sort_values("month")
    if len(g) < 6:
        return np.nan
    if g["month"].duplicated().any():
        raise ValueError("one observation per calendar month")
    months = pd.to_datetime(g["month"]).dt.to_period("M")
    x = months.astype("int64").to_numpy(dtype=float)
    x -= x.min()
    return float(stats.linregress(x, g["fpy"].to_numpy()).slope)



def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    engine = create_engine(get_settings().database_url)
    daily = pd.read_sql(DAILY_SQL, engine, parse_dates=["day"])
    monthly = pd.read_sql(
        MONTHLY_SQL, engine, parse_dates=["month"]
    )
    busy = daily.groupby("supplier_id")["day"].transform("size")
    return daily.loc[busy >= MIN_DAYS], monthly


def pareto(monthly: pd.DataFrame) -> list[str]:
    counts = (
        monthly.groupby("supplier_id")["ncr_count"]
               .sum()
               .sort_values(ascending=False)
    )
    if counts.empty or counts.sum() <= 0:
        return []
    cum = counts.cumsum() / counts.sum()
    n = int(cum.searchsorted(0.80)) + 1
    return counts.iloc[:n].index.tolist()


# Chapter 6 — Asher at Northlake (continued)
def build_report() -> str:
    daily, monthly = load()
    flagged = rolling_z(daily)
    anom = flagged.loc[
        (flagged["z"].abs() > Z_THRESHOLD)
        | flagged["flat_baseline_change"]
    ]
    trends = (
        monthly.sort_values(["supplier_id", "month"])
               .groupby("supplier_id")[["month", "fpy"]]
               .apply(fpy_slope)
    )
    degrading = trends[trends < SLOPE_THRESHOLD].sort_values()
    both = sorted(
        set(anom["supplier_id"]) & set(degrading.index)
    )
    log.info(
        "anomaly_report",
        flagged_days=len(anom),
        flagged_suppliers=anom["supplier_id"].nunique(),
        degrading=len(degrading),
        on_both=len(both),
    )
    return render(anom, degrading, both, date.today())


def render(anom, degrading, both, report_date) -> str:
    """Render measured flags without inventing business context."""
    lines = [
        f"# Supplier review report - {report_date}",
        "", "## Suppliers to ask about this week", "",
    ]
    lines.extend(f"- {sid}: on both review lists" for sid in both)
    if not both:
        lines.append("No supplier is on both lists.")
    lines += ["", "## Declining monthly FPY", ""]
    for sid, slope in degrading.items():
        lines.append(f"- {sid}: {slope:.4f} FPY per month")
    lines += ["", "## Unusual supplier days", ""]
    for row in anom.itertuples():
        detail = ("changed after a flat baseline"
                  if row.flat_baseline_change else f"z={row.z:.2f}")
        lines.append(f"- {row.supplier_id}, {row.day.date()}: {detail}")
    lines += ["", "Review open CARs, audits, and known process changes",
              "in the authorized quality system before acting."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(build_report())
