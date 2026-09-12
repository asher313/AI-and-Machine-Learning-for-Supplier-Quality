-- Teaching analytic schema: one row per CAR and per FINAL review, plus
-- adjudicated NCR recurrence events. These are separate from raw SAP extracts.
-- Parameter :asof is the frozen observation cutoff; dates after it are excluded.
WITH eligible AS (
  SELECT c.car_id, c.supplier_id, c.cause_code, c.closed_at, d.reviewer
  FROM sqm.car_outcomes c
  JOIN sqm.car_final_reviews d USING (car_id)
  WHERE c.closed_at + INTERVAL '6 months' <= :asof
    AND c.cause_code IS NOT NULL
), outcomes AS (
  SELECT c.*, EXISTS (
    SELECT 1 FROM sqm.ncr_recurrence_events r
    WHERE r.supplier_id = c.supplier_id
      AND r.cause_code = c.cause_code
      AND r.discovered_at > c.closed_at
      AND r.discovered_at <= c.closed_at + INTERVAL '6 months'
      AND r.adjudicated_at <= :asof
  ) AS recurred
  FROM eligible c
)
SELECT reviewer, count(*) AS eligible_cars,
       sum(recurred::int) AS cars_with_recurrence,
       avg(recurred::int) AS recurrence_rate
FROM outcomes GROUP BY reviewer;
