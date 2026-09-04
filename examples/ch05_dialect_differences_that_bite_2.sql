-- Chapter 5 — 5.2 Dialect Differences That Bite
-- Hints: read them, rarely write them
SELECT /*+ USE_HEX_PLAN */ supplier_id, COUNT(*)
  FROM SQM.NCRS GROUP BY supplier_id;
