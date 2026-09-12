-- Chapter 4 — 4.7 Reading an EXPLAIN Plan
-- B-tree index: the default; equality and range lookups
CREATE INDEX idx_ncrs_supplier_severity
  ON sqm.ncrs (supplier_id, severity);

-- Partial index: only the rows you actually query
CREATE INDEX idx_ncrs_high_severity
  ON sqm.ncrs (supplier_id)
  WHERE severity >= 3;

-- Expression index: for a function you filter by
CREATE INDEX idx_ncrs_month
  ON sqm.ncrs (DATE_TRUNC(
    'month', discovered_at AT TIME ZONE 'UTC'));
