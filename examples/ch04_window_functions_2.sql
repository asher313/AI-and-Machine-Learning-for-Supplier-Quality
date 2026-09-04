-- Chapter 4 — 4.3 Window Functions
-- Per-supplier average severity, and cumulative cost
SELECT ncr_id, supplier_id, discovered_at, severity,
       AVG(severity) OVER (
         PARTITION BY supplier_id
       ) AS supplier_avg_severity,
       SUM(cost_impact_usd) OVER (
         PARTITION BY supplier_id
         ORDER BY discovered_at
         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS cumulative_cost
FROM sqm.ncrs;
