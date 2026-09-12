-- PostgreSQL 16 with pgvector 0.8.6; initialize in a new teaching database.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE sqm.doc_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL,
    document_revision TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('as9100','manual','car','audit')),
    chunk_index INT NOT NULL CHECK (chunk_index >= 0),
    clause TEXT,
    content TEXT NOT NULL,
    token_count INT NOT NULL CHECK (token_count >= 0),
    token_model TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    embed_model TEXT NOT NULL,
    metadata JSONB NOT NULL,
    valid_from DATE NOT NULL,
    valid_to DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (valid_to IS NULL OR valid_to > valid_from),
    CHECK (metadata->>'visibility' IS NOT NULL AND
      ((metadata->>'visibility' = 'shared') OR
       (metadata->>'visibility' = 'program' AND metadata->>'program' IS NOT NULL))),
    UNIQUE(document_id, document_revision, chunk_index, embed_model)
);
CREATE INDEX doc_chunks_hnsw ON sqm.doc_chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX doc_chunks_program ON sqm.doc_chunks ((metadata->>'program'));
CREATE INDEX doc_chunks_scope ON sqm.doc_chunks (source_type, embed_model, valid_from, valid_to);
