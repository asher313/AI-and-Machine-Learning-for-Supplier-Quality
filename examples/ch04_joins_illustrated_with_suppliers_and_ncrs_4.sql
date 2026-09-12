-- Chapter 4 — 4.2 Joins, Illustrated With Suppliers and NCRs
-- Self-join: pairs of NCRs, same supplier, within 30 days
SELECT a.supplier_id,
       a.ncr_id AS first_ncr,
       b.ncr_id AS second_ncr
FROM sqm.ncrs a
JOIN sqm.ncrs b
  ON b.supplier_id = a.supplier_id
 AND (b.discovered_at, b.ncr_id)
       > (a.discovered_at, a.ncr_id)
 AND b.discovered_at <= a.discovered_at + INTERVAL '30 days'
ORDER BY a.supplier_id, a.discovered_at;
