-- Chapter 4 — 4.6 The SQM Query Cookbook
-- Pareto: the vital few suppliers behind 80% of NCRs
WITH supplier_totals AS (
  SELECT supplier_id, COUNT(*) AS ncr_count
  FROM sqm.ncrs
  WHERE discovered_at >= DATE '2025-09-01'
  GROUP BY supplier_id
),
ranked AS (
  SELECT supplier_id, ncr_count,
         SUM(ncr_count) OVER (
           ORDER BY ncr_count DESC, supplier_id
           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
         ) AS running_sum,
         SUM(ncr_count) OVER () AS total
  FROM supplier_totals
)
SELECT supplier_id, ncr_count,
       ROUND(100.0 * running_sum / total, 2) AS cumulative_pct
FROM ranked
WHERE running_sum - ncr_count < 0.8 * total
ORDER BY ncr_count DESC;
