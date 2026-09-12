# Build 3: offline pipeline walkthrough

From the repository root after installing the project:

```bash
python scripts/generate_triage_data.py
python -m sqm_ai.triage.run_build3
```

The generator writes 20 fictional records and scripted model responses to ignored
`data/triage/replay_cases.json`. The runner writes ignored
`artifacts/build3/decisions.jsonl`. Both refuse to overwrite an existing output.
This is a software replay: zero API calls, no database required, all 20 decisions
remain in the default review/shadow mode. It does not measure model quality.
No generated data is committed.

The live direct-API adapter validates structured tool responses, uses one retry
owner, traces every HTTP attempt, and rejects unknown/restricted data before
transmission. The source classification must come from trusted application
provenance. Regex detection is supplementary, not clearance or anonymization.
An approved provider/gateway adapter can be supplied through the pipeline's
classifier dependency. No supplied component releases parts, executes
suggested dispositions, sends email, or writes SAP.

The shipped rules supply partial proposals and conservative review triggers.
They do not reproduce the fictional 31% rules-only coverage. Rule/model
conflicts, high severity, missing ownership, low confidence, and default shadow
mode route to review. Rejected results remain absent rather than receiving
invented fields. Confidence cutoffs demonstrate policy mechanics; they require
independent evaluation before internal automatic assignment is enabled.

`triage/schema.sql` and `storage.write_decision` append attempts in PostgreSQL;
review events occupy a separate table. A new decision ID preserves re-triage of
the same NCR. Use a migration for an existing database. Production roles must
prevent unauthorized modification of audit rows and govern retained input,
prompt, policy, dependency, and corpus revisions. Full hashes help compare
snapshots but do not reconstruct or anonymize the original text.

Live evaluation is explicitly opt-in:

```bash
SQM_GOLDEN_NCRS=/approved/path/golden.json python -m pytest -m llm tests/triage/test_golden.py
```

Provide an independently adjudicated held-out corpus with `ncr`, `category`,
and `severity` per record. The generated replay fixture is not this corpus.
The input contract includes a trusted supplier shortlist and data classification.
Separate retrieval/development data from final evaluation. Report abstention,
category/severity errors, sample counts, and uncertainty; zero observed errors
on a finite safety subset does not establish zero future risk.

Validation performed September 11, 2026: 24 focused offline tests across the
Chapter 2 introduction and Chapters 15–16, one isolated PostgreSQL audit test,
and the 20-record replay. No paid model calls or production deployment tested.
