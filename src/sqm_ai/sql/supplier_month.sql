-- src/sqm_ai/sql/supplier_month.sql
DROP TABLE IF EXISTS sqm.supplier_month;
CREATE TABLE sqm.supplier_month AS
WITH months AS (
  SELECT generate_series(
           DATE '2025-09-01', DATE '2026-08-01',
           INTERVAL '1 month')::date AS month
),
spine AS (
  SELECT
    s.supplier_id, s.tier, m.month,
    (m.month + INTERVAL '1 month')::date AS month_end
  FROM sqm.suppliers s
  CROSS JOIN months m
  WHERE s.active
),
ncr AS (
  SELECT
    supplier_id,
    DATE_TRUNC('month', discovered_at)::date AS month,
    COUNT(*) AS ncr_count,
    SUM((severity >= 3)::int) AS sev3_plus_count,
    AVG(severity) AS avg_severity,
    SUM((category = 'dimensional')::int) AS dimensional_count,
    SUM((category = 'material')::int) AS material_count,
    SUM((category = 'cosmetic')::int) AS cosmetic_count,
    SUM((category = 'functional')::int) AS functional_count,
    COALESCE(SUM(cost_impact_usd), 0) AS cost_impact_usd
  FROM sqm.ncrs
  GROUP BY supplier_id, DATE_TRUNC('month', discovered_at)
),
insp AS (
  SELECT
    supplier_id,
    DATE_TRUNC('month', inspected_at)::date AS month,
    AVG(passed::int) AS fpy
  FROM sqm.inspections
  GROUP BY supplier_id, DATE_TRUNC('month', inspected_at)
),
deliv AS (
  SELECT
    p.supplier_id,
    DATE_TRUNC('month', g.received_at)::date AS month,
    AVG((g.received_at::date <= p.promised_date)::int) AS otd
  FROM sqm.goods_receipts g
  JOIN sqm.purchase_orders p ON p.po_line_id = g.po_line_id
  GROUP BY p.supplier_id, DATE_TRUNC('month', g.received_at)
)
SELECT
  sp.supplier_id, sp.tier, sp.month,
  COALESCE(n.ncr_count, 0)         AS ncr_count,
  COALESCE(n.sev3_plus_count, 0)   AS sev3_plus_count,
  n.avg_severity,
  COALESCE(n.dimensional_count, 0) AS dimensional_count,
  COALESCE(n.material_count, 0)    AS material_count,
  COALESCE(n.cosmetic_count, 0)    AS cosmetic_count,
  COALESCE(n.functional_count, 0)  AS functional_count,
  COALESCE(n.cost_impact_usd, 0)   AS cost_impact_usd,
  i.fpy,
  d.otd,
  (SELECT a.audit_score FROM sqm.audits a
    WHERE a.supplier_id = sp.supplier_id
      AND a.audit_date < sp.month_end
    ORDER BY a.audit_date DESC
    LIMIT 1) AS audit_score,
  (SELECT COUNT(*) FROM sqm.cars c
    WHERE c.supplier_id = sp.supplier_id
      AND c.opened_at < sp.month_end
      AND (c.closed_at IS NULL
           OR c.closed_at >= sp.month_end)) AS open_cars,
  SUM(COALESCE(n.ncr_count, 0)) OVER (
    PARTITION BY sp.supplier_id
    ORDER BY sp.month
    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
  ) AS ncr_count_3mo
FROM spine sp
LEFT JOIN ncr n
  ON n.supplier_id = sp.supplier_id AND n.month = sp.month
LEFT JOIN insp i
  ON i.supplier_id = sp.supplier_id AND i.month = sp.month
LEFT JOIN deliv d
  ON d.supplier_id = sp.supplier_id AND d.month = sp.month;
