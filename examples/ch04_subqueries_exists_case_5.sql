-- Chapter 4 — 4.5 Subqueries, EXISTS, CASE
-- Pivot: one column per NCR category, per supplier
SELECT supplier_id,
       SUM(CASE WHEN category = 'dimensional'
                THEN 1 ELSE 0 END) AS dimensional_count,
       SUM(CASE WHEN category = 'material'
                THEN 1 ELSE 0 END) AS material_count,
       SUM(CASE WHEN category = 'cosmetic'
                THEN 1 ELSE 0 END) AS cosmetic_count,
       SUM(CASE WHEN category = 'functional'
                THEN 1 ELSE 0 END) AS functional_count
FROM sqm.ncrs
WHERE discovered_at >= DATE '2025-09-01'
GROUP BY supplier_id;
