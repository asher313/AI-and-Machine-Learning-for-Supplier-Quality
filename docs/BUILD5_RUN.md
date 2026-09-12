# Build 5: runnable CAR drafting and review replay

Use the optional framework environment from `docs/AGENTS_RUN.md`. From the
repository root, generate the fictional input and run the draft:

```bash
python scripts/generate_car_data.py
python -m sqm_ai.car.run_build5
```

This creates `data/car/replay.json`, a SQLite checkpoint under
`artifacts/build5/`, and `artifacts/build5/draft.json`. The first run stops with
`awaiting_review`. Inspect its seven-section draft. An explicit synthetic review
can resume the same stored run, even from another process:

```bash
python -m sqm_ai.car.run_build5 --decision accept_draft \
  --output artifacts/build5/reviewed.json
```

Use `--decision reject` to exercise rejection. Existing output files and run IDs
are protected against accidental overwrite; use a new output or thread ID for
another run. This CLI accepts only generated synthetic bundles and a synthetic
reviewer. It is not an authentication or approval interface for company data.
Both steps were run successfully with zero API requests. The four model roles
use scripted outputs; research and assembly are code. No sample plan, causal
claim, confidence score, quality result, latency, or dollar cost is validated by
that replay. Its 50-unit/30-day plan is arbitrary software test data.

## What the implementation enforces

- A trusted `CarServices` instance binds the NCR and supplier, the approved
  source reader, outbound data classification, pseudonym map, role adapter,
  and reviewer authorization callback. A production service must authenticate
  the caller and enforce source program/date access before returning evidence.
- Research validates source IDs and scope, then pseudonymizes the model-bound
  evidence. Known aliases are case-insensitive; ambiguous aliases and reserved
  source tokens fail. Mappings must remain in protected application storage
  across resumes. Unknown aliases and indirect identity clues remain possible.
- Role outputs have runtime schemas. Root-cause links distinguish observations,
  inferences, and unknowns. Candidates are not confirmed physical causes.
  Unknown links require a concrete gap and cannot fabricate source references.
- A deterministic validator checks shape and reference membership. A role gets
  at most two repair attempts after its first call, with downstream sections
  recomputed. Multiple errors do not consume several retries at once.
- Every draft pauses for human review. Failures or explicit gaps block acceptance;
  reviewed replacement evidence returns the graph to research. Reviewer
  authorization must bind `actor_id` to the authenticated person. A string in a
  resume payload is not an identity credential.
- The SQLite checkpointer is held open with a context manager. Use a stable
  per-run thread ID and protect the checkpoint store: it contains draft/source
  data. Production storage, multiworker operation, crash recovery around writes,
  and a human UI need deployment-specific implementation.

Accepting a draft records a review outcome only. There is no supplier message,
SAP write, CAR closure, root-cause confirmation, or product-release operation.
The direct `call_role` and `analyze` adapters are available for explicitly
approved live inputs; they use the shared SDK boundary, request token checks,
structured output, complete-stop guards, bounded retries, and usage logging.
No live role-quality evaluation was performed.

## Memory and measurement

`CarMemory` is a separate optional component. It retains original note events,
scopes them to supplier/user/entitlements/policy revision, and stores derived
summaries with source-event IDs. The application must provide an approved
summarizer and substantiate any `confirmed_outcome` label. Changed permissions
or supplier scope do not inherit another scope's notes. Semantic recall uses
the reviewer's access and as-of date. Summaries are explicitly unverified;
record retention, corrections, token budgets, and source entailment still need
application controls. The local teaching transaction includes summarization;
use a job/locking design suited to production concurrency.

`edit_fraction` is normalized character edit distance after whitespace
normalization, not a factual-quality score. `sql/car_outcomes.sql` defines a
separate analytic schema for mature CAR cohorts, final reviewer identity, and
adjudicated recurrence events. `sql/car_recurrence.sql` uses `EXISTS`, so two
repeat events for one CAR contribute one positive outcome. It excludes immature
six-calendar-month windows, unadjudicated cause codes, and events adjudicated
after the evaluation cutoff. Report missing adjudication coverage separately;
compare shipment exposure and case mix before interpreting trends causally.

Validation: seven offline tests cover pseudonym collisions and restoration,
unknown-link coherence, persistent pause/resume, bounded repairs, new evidence,
unauthorized or gapped acceptance, and scoped persistent memory provenance.
One PostgreSQL test confirms the mature cohort denominator with duplicate
recurrence events, immature CARs, missing causes, and late adjudications.

Primary references checked September 11, 2026:

- [ASQ Five Whys](https://asq.org/quality-resources/five-whys)
- [NIST performance-threshold test design](https://www.nist.gov/publications/confirming-performance-threshold-binary-experimental-response)
- [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [RapidFuzz Levenshtein distance](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Levenshtein.html)
