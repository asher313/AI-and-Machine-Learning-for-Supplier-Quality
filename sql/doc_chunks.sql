-- sql/doc_chunks.sql
-- Chapter 17 — 17.5 Vector Stores
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE sqm.doc_chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  TEXT NOT NULL,
    source_type  TEXT NOT NULL,   -- as9100|manual|car|audit
    chunk_index  INT  NOT NULL,
    clause       TEXT,            -- '8.4.2' where it applies
    content      TEXT NOT NULL,
    token_count  INT  NOT NULL,
    embedding    VECTOR(1024) NOT NULL,
    embed_model  TEXT NOT NULL,
    metadata     JSONB NOT NULL DEFAULT '{}',
    valid_from   DATE NOT NULL,
    valid_to     DATE,            -- NULL = current
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT source_enum CHECK (source_type IN
        ('as9100','manual','car','audit'))
);

-- Approximate nearest neighbour, cosine distance.
CREATE INDEX doc_chunks_hnsw
    ON sqm.doc_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Metadata filters (program, supplier_id, ...).
CREATE INDEX doc_chunks_meta
    ON sqm.doc_chunks USING GIN (metadata);

CREATE INDEX doc_chunks_type_current
    ON sqm.doc_chunks (source_type)
    WHERE valid_to IS NULL;
