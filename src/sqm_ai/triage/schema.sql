-- Append-only attempts. Apply to a new teaching database, not over an existing table.
CREATE TABLE sqm.ncr_classifications (
    decision_id        UUID PRIMARY KEY,
    ncr_id             VARCHAR(32) NOT NULL,
    source_commit_sha  VARCHAR(40) NOT NULL,
    classifier_version VARCHAR(32) NOT NULL,
    model              VARCHAR(64) NOT NULL,
    stage              VARCHAR(32) NOT NULL,
    category           VARCHAR(32),
    severity           SMALLINT,
    supplier_id        VARCHAR(32),
    suggested_disposition VARCHAR(32),
    car_owner_email    VARCHAR(128),
    confidence         REAL,
    reasoning          TEXT,
    rules_fired        TEXT[] NOT NULL DEFAULT '{}',
    trace_ids          UUID[] NOT NULL DEFAULT '{}',
    classified_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    routed             VARCHAR(16) NOT NULL,
    review_notes       TEXT,
    CONSTRAINT complete_result CHECK (
        (category IS NULL AND severity IS NULL AND supplier_id IS NULL
         AND confidence IS NULL AND reasoning IS NULL AND suggested_disposition IS NULL)
        OR (category IS NOT NULL AND severity IS NOT NULL AND supplier_id IS NOT NULL
         AND confidence IS NOT NULL AND reasoning IS NOT NULL AND suggested_disposition IS NOT NULL)),
    CONSTRAINT category_enum CHECK (category IN ('cosmetic','dimensional','material','functional')),
    CONSTRAINT severity_range CHECK (severity BETWEEN 1 AND 5),
    CONSTRAINT confidence_range CHECK (confidence BETWEEN 0 AND 1),
    CONSTRAINT disposition_enum CHECK (suggested_disposition IN ('use-as-is','rework','scrap','return-to-supplier')),
    CONSTRAINT routed_enum CHECK (routed IN ('auto','review')),
    CONSTRAINT auto_has_result CHECK (routed <> 'auto' OR
       (category IS NOT NULL AND severity <= 3 AND confidence >= .90 AND car_owner_email IS NOT NULL))
);
CREATE INDEX idx_ncr_class_record ON sqm.ncr_classifications(ncr_id, classified_at DESC);
CREATE INDEX idx_ncr_class_review ON sqm.ncr_classifications(classified_at) WHERE routed = 'review';
CREATE TABLE sqm.ncr_classification_reviews (
    review_id UUID PRIMARY KEY,
    decision_id UUID NOT NULL REFERENCES sqm.ncr_classifications(decision_id),
    reviewed_by VARCHAR(128) NOT NULL,
    reviewed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    review_decision VARCHAR(16) NOT NULL CHECK (review_decision IN ('confirmed','modified','overridden')),
    corrected_result JSONB,
    review_notes TEXT,
    CONSTRAINT correction_required CHECK (review_decision = 'confirmed' OR corrected_result IS NOT NULL)
);
-- Probe task records belong in a separate controlled evaluation store.
-- Production roles must restrict UPDATE/DELETE and protect the audit/trace stores.
