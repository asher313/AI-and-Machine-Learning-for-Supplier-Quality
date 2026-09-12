# Build 6: synthetic gateway replay

This text-only adapter demonstrates trusted classification, full request screening, an explicit approved endpoint registry, separately committed audit events, encrypted payload references, output withholding, and atomic estimated-spend reservations. It does not establish regulatory compliance or automatically intercept the earlier chapters' direct SDK calls. Deployments must migrate every relevant call and enforce identity, credential and network boundaries.

Install the project (`uv sync`) and use a disposable PostgreSQL database whose role can create schema objects. The replay adds records; it does not drop or truncate existing tables. Never point the exercise at production.

```bash
uv run python scripts/run_build6.py --dsn 'postgresql://USER@localhost/DATABASE' --out work/build6-replay
```

The generator constructs 33 phone-number blocks, eight misplaced-marking blocks, and one allowed call. These reproduce the chapter's fictional 33/8 descriptive counts; they measure no real-world detection rate. Only the one allowed request reaches an in-process scripted endpoint. There are zero provider API calls. Its $0.02 quote and $0.01 charge are arbitrary accounting fixtures, not model prices. Generated reports and encrypted blobs remain under ignored `work/`.

The replay verifies encrypted request reconstruction before exit. Its random key exists only in memory: the resulting blobs cannot be reopened afterwards. This is a software demonstration, not a seven-year archive. A deployed store requires approved key retention/rotation, access controls, backup, deletion and legal-hold procedures. AES-GCM here is not a claim of FIPS validation or permission to process export-controlled data. Metadata and hashes also require access controls.

Run the behavioral tests with an isolated database creator role:

```bash
SQM_TEST_DATABASE_URL='postgresql+psycopg://USER@localhost/postgres' uv run pytest tests/gateway
```

Each SQL test creates and drops its own UUID-named database. Tests cover full context screening, unknown classification, output citation bounds, incomplete responses, higher-marked output storage, encryption tampering and wrong keys, audit outages, provider timeouts, concurrent reservations, cap consistency and explicit reconciliation. The phrase/PII detectors remain incomplete heuristics; valid citation numbers do not establish evidence support.

An operator registers each endpoint with its actual provider model ID, approved classifications, a conservative Decimal quote function and a usage-based charge function for the specific account/rates. The endpoint must disable hidden retries. Include every billable component and a maximum output allowance in the estimate; unsupported pricing must prevent dispatch. The sample does not infer approval from `controlled` or manufacture Bedrock model IDs.

The ledger locks a UTC calendar-month row before reserving, rounds amounts upward to eight decimal places, and refuses reservations that exceed the shared cap. Different cap settings for the same month fail closed. Policy cap changes require an explicit operator migration. A known charge above its quote is recorded and the response withheld; therefore this is an estimated-spend guard, not an invoice guarantee. Upstream failures retain an uncertain reservation. After checking provider billing evidence, an authorized operator may call `ledger.settle(trace_id, verified_amount, reconcile=True)`; persist that evidence in the organization's accounting audit. Never release a timeout reservation merely because no response arrived. Repeated acknowledgement of an identical settled charge is idempotent.

`received` and `dispatch_intent` events precede execution; final events precede display. A process crash can leave a pending intent, and independent SQL/content/provider transactions cannot guarantee exactly-once execution or a final event. Reconcile pending traces. Audit outages suppress output; no sink can record a request during its own outage. Blocked request text is hashed but not archived. A response requiring a higher approved archive is withheld and recorded as an archive gap. Operational investigations need an approved escalation process for these gaps. Logical request/response reconstruction does not recreate provider internals or guarantee deterministic reruns.
