-- src/sqm_ai/build1/sql/labels.sql
-- Bind data_complete_through to a verified exclusive watermark.
WITH cutoffs AS (
  SELECT supplier_id, month,
         month + INTERVAL '1 month' AS cutoff,
         month + INTERVAL '1 month' + INTERVAL '90 days'
           AS label_end
  FROM sqm.supplier_month
)
SELECT sp.supplier_id, sp.month, sp.label_end,
  CAST(:data_complete_through AS timestamp)
    AS data_complete_through,
  CASE WHEN sp.label_end <= CAST(:data_complete_through AS timestamp)
         AND COUNT(n.ncr_id) = COUNT(n.severity)
       THEN COALESCE(MAX((n.severity >= 3)::int), 0)
  END AS sev3_next_90d,
  CASE WHEN sp.label_end <= CAST(:data_complete_through AS timestamp)
         AND COUNT(n.ncr_id) = COUNT(n.severity)
       THEN COALESCE(SUM(n.severity), 0)
  END AS harm_next_90d
FROM cutoffs sp
LEFT JOIN sqm.ncrs n
  ON n.supplier_id = sp.supplier_id
 AND n.discovered_at >= sp.cutoff
 AND n.discovered_at < sp.label_end
GROUP BY sp.supplier_id, sp.month, sp.label_end;
