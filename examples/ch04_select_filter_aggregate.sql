-- Chapter 4 — 4.1 SELECT, Filter, Aggregate
-- Pattern 1: project, filter, order, limit
SELECT ncr_id, supplier_id, part_number, severity
FROM sqm.ncrs
WHERE severity >= 4
  AND discovered_at >= DATE '2026-06-01'
ORDER BY discovered_at DESC
LIMIT 50;
