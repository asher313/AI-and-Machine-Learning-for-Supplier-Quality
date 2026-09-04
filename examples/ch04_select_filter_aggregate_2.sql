-- Chapter 4 — 4.1 SELECT, Filter, Aggregate
-- Pattern 2: aggregate per group, then filter the groups
SELECT supplier_id,
       COUNT(*)             AS ncr_count,
       AVG(severity)        AS avg_severity,
       MAX(cost_impact_usd) AS worst_cost
FROM sqm.ncrs
WHERE discovered_at >= DATE '2025-09-01'
GROUP BY supplier_id
HAVING COUNT(*) >= 5
ORDER BY ncr_count DESC;
