# Chapter 5 — 5.4 Python to HANA
from sqm_ai.hana import read_hana

SQL = """
SELECT supplier_id, SUM(cost_impact_usd) AS total_cost
  FROM SQM.NCRS
 WHERE discovered_at >= ?
 GROUP BY supplier_id
 ORDER BY total_cost DESC
"""

df = read_hana(SQL, ("2025-09-01",))
print(df.head())
