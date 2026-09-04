-- src/sqm_ai/build1/sql/features.sql
CREATE OR REPLACE VIEW sqm.supplier_features AS
SELECT
  sm.supplier_id, sm.tier, sm.month,
  s.onboarded_at,
  SUM(sm.ncr_count)       OVER w3 AS ncr_count_90d,
  SUM(sm.sev3_plus_count) OVER w3 AS sev3_count_90d,
  AVG(sm.avg_severity)    OVER w3 AS avg_severity_90d,
  SUM(sm.units_received)  OVER w3 AS units_received_90d,
  SUM(sm.spend_usd)       OVER w3 AS spend_90d,
  AVG(sm.otd)             OVER w3 AS otd_pct,
  AVG(sm.days_late)       OVER w3 AS avg_days_late,
  sm.fpy,
  sm.audit_score                  AS audit_score_last,
  sm.audit_date                   AS audit_date_last,
  sm.car_response_days,
  sm.car_effectiveness,
  sm.open_cars
FROM sqm.supplier_month sm
JOIN sqm.suppliers s USING (supplier_id)
WINDOW w3 AS (
  PARTITION BY sm.supplier_id ORDER BY sm.month
  ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
);
