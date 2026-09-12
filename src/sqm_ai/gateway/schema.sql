-- Separate append-only lifecycle events; default partition prevents rollover gaps.
CREATE TABLE IF NOT EXISTS sqm.llm_audit_events (
  trace_id uuid NOT NULL,
  ts timestamptz NOT NULL DEFAULT now(),
  event text NOT NULL,
  record jsonb NOT NULL,
  PRIMARY KEY (trace_id, ts, event)
) PARTITION BY RANGE (ts);
CREATE TABLE IF NOT EXISTS sqm.llm_audit_events_default
  PARTITION OF sqm.llm_audit_events DEFAULT;
CREATE INDEX IF NOT EXISTS llm_audit_trace ON sqm.llm_audit_events(trace_id,ts);
CREATE TABLE IF NOT EXISTS sqm.gateway_budget_months (
  period date PRIMARY KEY,
  cap numeric(18,8) NOT NULL CHECK(cap>0),
  spent numeric(18,8) NOT NULL DEFAULT 0 CHECK (spent>=0),
  reserved numeric(18,8) NOT NULL DEFAULT 0 CHECK (reserved>=0)
);
CREATE TABLE IF NOT EXISTS sqm.gateway_reservations (
  trace_id uuid PRIMARY KEY,
  period date NOT NULL REFERENCES sqm.gateway_budget_months(period),
  quoted numeric(18,8) NOT NULL CHECK (quoted>=0),
  actual numeric(18,8),
  status text NOT NULL CHECK(status IN ('reserved','settled','uncertain'))
);
