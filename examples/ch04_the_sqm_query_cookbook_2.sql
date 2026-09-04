-- Chapter 4 — 4.6 The SQM Query Cookbook
-- First-pass yield per supplier per month
SELECT supplier_id,
       DATE_TRUNC('month', inspected_at)::date AS month,
       COUNT(*) AS units,
       ROUND(AVG(passed::int), 4) AS fpy
FROM sqm.inspections
WHERE inspected_at >= DATE '2025-09-01'
GROUP BY supplier_id, DATE_TRUNC('month', inspected_at)
ORDER BY supplier_id, month;
