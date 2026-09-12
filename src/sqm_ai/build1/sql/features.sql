-- src/sqm_ai/build1/sql/features.sql
-- Optional, independently governed source features. Absence stays NULL.
CREATE TABLE IF NOT EXISTS sqm.supplier_feature_enrichment (
  supplier_id text NOT NULL,
  cutoff date NOT NULL,
  spend_90d double precision,
  car_response_days double precision,
  car_effectiveness double precision,
  PRIMARY KEY (supplier_id, cutoff)
);

CREATE OR REPLACE VIEW sqm.supplier_features AS
SELECT
  sm.supplier_id, sm.tier, sm.month, s.onboarded_at,
  n.ncr_count_90d, n.sev3_count_90d, n.avg_severity_90d,
  r.units_received_90d, e.spend_90d,
  r.otd_pct, r.avg_days_late, sm.fpy,
  a.audit_score AS audit_score_last,
  a.audit_date AS audit_date_last,
  e.car_response_days, e.car_effectiveness, sm.open_cars
FROM sqm.supplier_month sm
JOIN sqm.suppliers s USING (supplier_id)
CROSS JOIN LATERAL (
  SELECT sm.month + INTERVAL '1 month' AS cutoff
) c
CROSS JOIN LATERAL (
  SELECT COUNT(*) AS ncr_count_90d,
    COUNT(*) FILTER (WHERE severity >= 3) AS sev3_count_90d,
    AVG(severity) AS avg_severity_90d
  FROM sqm.ncrs n
  WHERE n.supplier_id = sm.supplier_id
    AND n.discovered_at >= c.cutoff - INTERVAL '90 days'
    AND n.discovered_at < c.cutoff
) n
CROSS JOIN LATERAL (
  SELECT CASE WHEN COUNT(*) = 0 THEN 0
              WHEN COUNT(g.quantity) = COUNT(*)
              THEN SUM(g.quantity) END AS units_received_90d,
    CASE WHEN COUNT(p.promised_date) = COUNT(*) THEN
      AVG((g.received_at::date <= p.promised_date)::int)
    END AS otd_pct,
    CASE WHEN COUNT(p.promised_date) = COUNT(*) THEN
      AVG(GREATEST(g.received_at::date - p.promised_date, 0))
    END AS avg_days_late
  FROM sqm.goods_receipts g
  JOIN sqm.purchase_orders p ON p.po_line_id = g.po_line_id
  WHERE g.supplier_id = sm.supplier_id
    AND g.received_at >= c.cutoff - INTERVAL '90 days'
    AND g.received_at < c.cutoff
) r
LEFT JOIN LATERAL (
  SELECT audit_score, audit_date FROM sqm.audits a
  WHERE a.supplier_id = sm.supplier_id
    AND a.audit_date < c.cutoff
  ORDER BY audit_date DESC, audit_id DESC LIMIT 1
) a ON TRUE
LEFT JOIN sqm.supplier_feature_enrichment e
  ON e.supplier_id = sm.supplier_id
 AND e.cutoff = c.cutoff::date;
