-- Chapter 4 — 4.5 Subqueries, EXISTS, CASE
-- Suppliers with at least one severity >= 4 NCR since Sept
SELECT s.supplier_id, s.supplier_name
FROM sqm.suppliers s
WHERE EXISTS (
  SELECT 1
  FROM sqm.ncrs n
  WHERE n.supplier_id = s.supplier_id
    AND n.severity >= 4
    AND n.discovered_at >= DATE '2025-09-01'
);
