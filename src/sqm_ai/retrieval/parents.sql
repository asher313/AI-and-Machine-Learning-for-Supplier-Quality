CREATE TABLE sqm.doc_parents (
    id BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL,
    document_revision TEXT NOT NULL,
    source_type TEXT NOT NULL,
    metadata JSONB NOT NULL,
    valid_from DATE NOT NULL,
    valid_to DATE,
    content TEXT NOT NULL,
    UNIQUE(id, document_id, document_revision),
    CHECK (valid_to IS NULL OR valid_to > valid_from)
);
ALTER TABLE sqm.doc_chunks ADD COLUMN parent_id BIGINT;
ALTER TABLE sqm.doc_chunks ADD CONSTRAINT parent_same_revision
    FOREIGN KEY(parent_id, document_id, document_revision)
    REFERENCES sqm.doc_parents(id, document_id, document_revision);
