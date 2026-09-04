-- src/sqm_ai/triage/schema.sql  (Postgres, schema sqm)
CREATE TABLE sqm.ncr_classifications (
    ncr_id             VARCHAR(32) PRIMARY KEY,
    source_commit_sha  VARCHAR(40)  NOT NULL,
    classifier_version VARCHAR(16)  NOT NULL,
    model              VARCHAR(64)  NOT NULL,
    stage              VARCHAR(16)  NOT NULL,
    category           VARCHAR(32)  NOT NULL,
    severity           SMALLINT     NOT NULL,
    supplier_id        VARCHAR(32),
    car_owner_email    VARCHAR(128),
    confidence         REAL         NOT NULL,
    reasoning          TEXT,
    rules_fired        TEXT[]  NOT NULL DEFAULT '{}',
    trace_id           UUID         NOT NULL,
    classified_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    routed             VARCHAR(16)  NOT NULL,
    is_known_answer    BOOLEAN NOT NULL DEFAULT FALSE,
    reviewed_by        VARCHAR(128),
    reviewed_at        TIMESTAMPTZ,
    review_decision    VARCHAR(16),
    review_notes       TEXT,
    CONSTRAINT category_enum CHECK (category IN
        ('cosmetic','dimensional','material','functional')),
    CONSTRAINT severity_range
        CHECK (severity BETWEEN 1 AND 5),
    CONSTRAINT confidence_range
        CHECK (confidence BETWEEN 0 AND 1),
    CONSTRAINT routed_enum
        CHECK (routed IN ('auto','review')),
    CONSTRAINT decision_enum CHECK (review_decision IS NULL
        OR review_decision IN
           ('confirmed','modified','overridden'))
);

CREATE INDEX idx_ncr_class_unreviewed
    ON sqm.ncr_classifications (classified_at)
    WHERE reviewed_at IS NULL;

CREATE INDEX idx_ncr_class_supplier_severity
    ON sqm.ncr_classifications (supplier_id, severity);
