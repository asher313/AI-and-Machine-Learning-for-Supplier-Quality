-- Chapter 4 — 4.2 Joins, Illustrated With Suppliers and NCRs
-- INNER: only NCRs whose supplier exists in the master
SELECT n.ncr_id, s.supplier_name, s.tier
FROM sqm.ncrs n
INNER JOIN sqm.suppliers s ON s.supplier_id = n.supplier_id;
