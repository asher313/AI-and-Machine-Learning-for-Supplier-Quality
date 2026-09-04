-- Chapter 4 — 4.4 CTEs and Recursion
-- Monthly counts -> rolling trend -> last month's ranking
WITH monthly AS (
  SELECT supplier_id,
         DATE_TRUNC('month', discovered_at)::date AS month,
         COUNT(*)      AS ncr_count,
         AVG(severity) AS avg_severity
  FROM sqm.ncrs
  WHERE discovered_at >= CURRENT_DATE - INTERVAL '12 months'
  GROUP BY supplier_id, DATE_TRUNC('month', discovered_at)
),
with_trend AS (
  SELECT *,
         AVG(ncr_count) OVER (
           PARTITION BY supplier_id
           ORDER BY month
           ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
         ) AS rolling_3mo,
         LAG(ncr_count) OVER (
           PARTITION BY supplier_id ORDER BY month
         ) AS prev_month
  FROM monthly
)
SELECT supplier_id, month, ncr_count, rolling_3mo,
       ncr_count - prev_month AS mom_delta
FROM with_trend
WHERE month = DATE_TRUNC('month', CURRENT_DATE)::date
              - INTERVAL '1 month'
ORDER BY rolling_3mo DESC;
