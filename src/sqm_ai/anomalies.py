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
    cum = counts.cumsum() / counts.sum()
    return cum[cum <= 0.80].index.tolist()


# Chapter 6 — Asher at Northlake (continued)
def build_report() -> str:
    daily, monthly = load()
    flagged = rolling_z(daily)
    anom = flagged.loc[flagged["z"].abs() > Z_THRESHOLD]
    trends = (
        monthly.sort_values(["supplier_id", "month"])
               .groupby("supplier_id")[["fpy"]]
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
