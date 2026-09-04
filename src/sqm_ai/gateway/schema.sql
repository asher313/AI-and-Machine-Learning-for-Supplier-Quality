-- sqm_ai/gateway/schema.sql   (Postgres)
CREATE TABLE llm_audit (
    trace_id        UUID        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    user_id         VARCHAR(128) NOT NULL,
    tool_name       VARCHAR(64)  NOT NULL,
    model           VARCHAR(64),
    -- enclave: open | internal | controlled
    enclave         VARCHAR(16)  NOT NULL,
    prompt_hash     CHAR(64)     NOT NULL, -- sha256, not text
    prompt_tokens   INT,
    response_hash   CHAR(64),
    response_tokens INT,
    pre_warnings    JSONB,
    post_warnings   JSONB,
    blocked         BOOLEAN      NOT NULL DEFAULT FALSE,
    block_codes     TEXT[],
    latency_ms      INT,
    cost_usd        NUMERIC(10, 6),
    user_feedback   VARCHAR(16),           -- good | bad | NULL
    PRIMARY KEY (trace_id, ts)
) PARTITION BY RANGE (ts);

-- one partition per month; a job creates next month's ahead
CREATE TABLE llm_audit_2026_09 PARTITION OF llm_audit
    FOR VALUES FROM ('2026-09-01') TO ('2026-10-01');

CREATE INDEX ON llm_audit (user_id, ts DESC);
CREATE INDEX ON llm_audit (blocked, ts DESC) WHERE blocked;
