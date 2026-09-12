-- Chapter 4 — 4.3 Window Functions
-- Rolling 30-day first-pass yield, per supplier, per day
WITH daily AS (
  SELECT supplier_id,
         inspected_at::date AS day,
         COUNT(*)          AS n,
         SUM(passed::int)  AS n_pass
  FROM sqm.inspections
  GROUP BY supplier_id, inspected_at::date
)
SELECT supplier_id, day, n,
       SUM(n_pass) OVER w * 1.0 / SUM(n) OVER w AS fpy_30d
FROM daily
WINDOW w AS (
  PARTITION BY supplier_id
  ORDER BY day
  RANGE BETWEEN INTERVAL '29 days' PRECEDING AND CURRENT ROW
)
ORDER BY supplier_id, day;
