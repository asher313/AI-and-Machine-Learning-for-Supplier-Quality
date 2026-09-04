-- Chapter 4 — 4.2 Joins, Illustrated With Suppliers and NCRs
-- LEFT: every supplier, with 0 for the quiet ones
SELECT s.supplier_id, s.supplier_name,
       COUNT(n.ncr_id) AS ncr_count
FROM sqm.suppliers s
LEFT JOIN sqm.ncrs n ON n.supplier_id = s.supplier_id
WHERE s.active
GROUP BY s.supplier_id, s.supplier_name
ORDER BY ncr_count DESC;
