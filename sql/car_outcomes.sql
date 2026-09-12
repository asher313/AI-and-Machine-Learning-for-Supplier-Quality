-- One analytic outcome per CAR, one final review, and one adjudicated event per NCR.
CREATE TABLE IF NOT EXISTS sqm.car_outcomes (
  car_id text PRIMARY KEY,
  supplier_id text NOT NULL,
  cause_code text,
  closed_at timestamp NOT NULL
);
CREATE TABLE IF NOT EXISTS sqm.car_final_reviews (
  car_id text PRIMARY KEY REFERENCES sqm.car_outcomes(car_id),
  reviewer text NOT NULL
);
CREATE TABLE IF NOT EXISTS sqm.ncr_recurrence_events (
  ncr_id text PRIMARY KEY,
  supplier_id text NOT NULL,
  cause_code text NOT NULL,
  discovered_at timestamp NOT NULL,
  adjudicated_at timestamp NOT NULL,
  CHECK (adjudicated_at >= discovered_at)
);
