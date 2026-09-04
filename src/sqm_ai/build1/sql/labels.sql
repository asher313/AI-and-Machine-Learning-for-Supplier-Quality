-- src/sqm_ai/build1/sql/labels.sql
-- Both targets look strictly forward from month end.
SELECT
  sp.supplier_id,
  sp.month,
  COALESCE(MAX((n.severity >= 3)::int), 0)
    AS sev3_next_90d,
  COALESCE(SUM(n.severity), 0) AS harm_next_90d
FROM sqm.supplier_month sp
LEFT JOIN sqm.ncrs n
  ON n.supplier_id = sp.supplier_id
 AND n.discovered_at >= (sp.month
       + INTERVAL '1 month')
 AND n.discovered_at <  (sp.month
       + INTERVAL '1 month' + INTERVAL '90 days')
GROUP BY sp.supplier_id, sp.month;
