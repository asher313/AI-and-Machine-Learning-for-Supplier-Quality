-- sql/doc_parents.sql
-- Chapter 17 — 17.9 Advanced Patterns
CREATE TABLE sqm.doc_parents (
    id      BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL          -- ~1,500 tokens
);
ALTER TABLE sqm.doc_chunks
    ADD COLUMN parent_id BIGINT
        REFERENCES sqm.doc_parents(id);
