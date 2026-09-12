-- Chapter 5 — 5.2 Dialect Differences That Bite
-- Hints: read them, rarely write them
SELECT supplier_id, COUNT(*)
  FROM SQM.NCRS GROUP BY supplier_id
  WITH HINT (USE_HEX_PLAN);
