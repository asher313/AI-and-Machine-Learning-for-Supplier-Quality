-- Chapter 4 — 4.5 Subqueries, EXISTS, CASE
-- Active suppliers with zero NCRs in 2026
SELECT s.supplier_id, s.supplier_name
FROM sqm.suppliers s
WHERE s.active
  AND NOT EXISTS (
    SELECT 1 FROM sqm.ncrs n
    WHERE n.supplier_id = s.supplier_id
      AND n.discovered_at >= DATE '2026-01-01'
  );
