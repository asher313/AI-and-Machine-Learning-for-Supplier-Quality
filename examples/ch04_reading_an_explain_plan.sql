-- Chapter 4 — 4.7 Reading an EXPLAIN Plan
EXPLAIN ANALYZE
SELECT ncr_id, severity
FROM sqm.ncrs
WHERE supplier_id = 'S-0417' AND severity >= 3;
