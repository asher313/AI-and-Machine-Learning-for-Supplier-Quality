-- Chapter 4 — 4.2 Joins, Illustrated With Suppliers and NCRs
-- FULL OUTER: suppliers with no NCRs AND NCRs with no supplier
SELECT s.supplier_id, n.ncr_id
FROM sqm.suppliers s
FULL OUTER JOIN sqm.ncrs n ON n.supplier_id = s.supplier_id
WHERE s.supplier_id IS NULL OR n.ncr_id IS NULL;
