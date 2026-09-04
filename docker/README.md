# docker/

```bash
docker compose -f docker/docker-compose.yml up -d
psql "postgresql://sqm:sqm@localhost:5432/sqm_analytics" \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Then point `DATABASE_URL` in `.env` at it:

```
DATABASE_URL=postgresql+psycopg://sqm:sqm@localhost:5432/sqm_analytics
```

`sql/doc_chunks.sql` (Chapter 17.5) creates the chunk table and
its HNSW index; `src/sqm_ai/sql/supplier_month.sql` (Chapter 4)
creates the `sqm` schema and the feature table;
`src/sqm_ai/triage/schema.sql` (Chapter 16.8) and
`src/sqm_ai/gateway/schema.sql` (Chapter 21.5) add Build 3's and
Build 6's tables.

Chapter 22.8 runs the same database as a GitLab service
container, which is where the `integration`-marked tests
written in Chapter 4 actually run.
