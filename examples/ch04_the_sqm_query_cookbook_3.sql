-- Chapter 4 — 4.6 The SQM Query Cookbook
-- Did the audit reduce NCRs? 90 days before vs after
WITH audit_windows AS (
  SELECT supplier_id, audit_date, audit_score,
         audit_date - INTERVAL '90 days' AS before_start,
         audit_date + INTERVAL '90 days' AS after_end
  FROM sqm.audits
  WHERE audit_date <= CURRENT_DATE - INTERVAL '90 days'
)
SELECT a.supplier_id, a.audit_date, a.audit_score,
       (SELECT COUNT(*) FROM sqm.ncrs n
         WHERE n.supplier_id = a.supplier_id
           AND n.discovered_at >= a.before_start
           AND n.discovered_at <  a.audit_date) AS before_count,
       (SELECT COUNT(*) FROM sqm.ncrs n
         WHERE n.supplier_id = a.supplier_id
           AND n.discovered_at >= a.audit_date
           AND n.discovered_at <  a.after_end) AS after_count
FROM audit_windows a
ORDER BY a.audit_date DESC;
