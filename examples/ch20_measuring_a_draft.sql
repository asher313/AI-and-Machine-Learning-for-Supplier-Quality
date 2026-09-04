-- Chapter 20 — 20.7 Measuring a Draft
-- share of closed CARs followed by a repeat within 6 months
WITH closed AS (
    SELECT car_id, supplier_id, root_cause, closed_at
      FROM sqm.cars
     WHERE status = 'closed'
       AND closed_at < current_date - INTERVAL '6 months'
)
SELECT d.reviewer,
       count(*) AS cars,
       avg((r.car_id IS NOT NULL)::int) AS recurrence_rate
  FROM closed c
  JOIN sqm.car_drafts d ON d.car_id = c.car_id
  LEFT JOIN sqm.cars r
    ON  r.supplier_id = c.supplier_id
    AND r.root_cause  = c.root_cause
    AND r.opened_at > c.closed_at
    AND r.opened_at <= c.closed_at + INTERVAL '6 months'
 GROUP BY d.reviewer;
