# Running Build 1 locally

Generate the synthetic data first (see [SYNTHETIC_DATA.md](SYNTHETIC_DATA.md)),
then run from the repository root:

```bash
uv sync --locked --extra ml
uv run --no-sync python -m sqm_ai.build1.train --input data/supplier_month.parquet --output models/build1
uv run --no-sync python -m sqm_ai.build1.score_suppliers --as-of 2026-09-01 --models models/build1 --data data/supplier_month_all.parquet --output data/supplier_risk_scores.parquet
```

Use new model/output paths for subsequent runs; commands protect existing
artifacts. A quick `--trees 30` training check is available, but its metrics
are not a default-model benchmark. On macOS, install OpenMP (`brew install
libomp`) if XGBoost reports a missing `libomp.dylib`.

The model predicts from a completed monthly snapshot. A later publication
date in that month republishes the same forecast window; it does not turn
this into a daily model. The final synthetic snapshot supports September
2026 publication. The scorer rejects stale snapshots and model artifacts
trained with data unavailable at the requested publication date.

## Model card: synthetic demonstration

- Intended use: instruction, execution checks, and human-reviewed audit planning.
- Inputs: 14 named numeric predictors plus supplier tier; targets and metadata
  are explicitly excluded. Source measures other than NCR history are constructed
  aggregate fixtures, not a full reconstruction of SAP transactions.
- Targets: any severity-3+ NCR and the severity sum in the next 90 days. Labels
  whose full window is unavailable remain missing and are excluded from training.
- Development evaluation: five chronological folds, a three-month maturity gap,
  and an inner three-month calibration period. The classifier and regressor
  have independent fitted preprocessing. The first inner fit can have no
  observed 12-month slope; the median imputer drops that column for that fit,
  and the fitted pipeline applies the same transformation at prediction time.
  Folds used to compare algorithms, features or hyperparameters are development
  evidence. They are not an untouched final evaluation after that selection.
- Scoring: floor predicted harm at zero; retain the raw prediction and clipping
  flag. SHAP values explain the raw regressor, not the floor, cohort percentile,
  or calibrated classifier probability. Percentiles use midpoint ranks for ties.
- Escalation: the illustrative probability threshold is 0.30. Limited history
  and insufficient activity suppress escalation; this is a teaching policy,
  not a validated operational recommendation or an uncertainty interval.
- Provenance: metadata contains data, source, and artifact SHA-256 hashes.
  Joblib artifacts must come from a trusted source; checksum agreement alone
  does not establish trust. Preserve the compatible dependency environment.
- Limitations: constructed data may encode artificial separability and temporal
  patterns. Scores do not establish real-world discrimination, calibration,
  fairness, or cost effectiveness. Late source corrections need availability-time
  controls and immutable snapshots in a real system.

Actual default-model and MLP runs are recorded in
[synthetic_build1_mlp_run.json](synthetic_build1_mlp_run.json).
They are intentionally separate from the textbook's fictional design tables.

The PostgreSQL adapter requires the source schemas described in Chapter 4,
the event tables, and any optional enrichment data. Numeric fixture generation
does not load those tables. Integration tests use `SQM_TEST_DATABASE_URL` to
create and remove isolated test databases; they do not use the application DB.

## Evaluate the actual monthly audit policy

Training now also saves `development_predictions.parquet` and
`monthly_audit_policy.json`. The policy ranks active supplier-month rows by
nonnegative predicted severity burden, breaking ties by ascending supplier ID,
and selects up to 50 each month. The same eligible cohort and tie rule apply to
the baseline: recent NCR count multiplied by recent mean severity. Zero NCRs
imply zero recent burden. Limited-history suppliers remain visible candidates;
that flag affects the separate automatic escalation rule, not this audit list.

For each complete month and policy, the report retains selected IDs, eligible
and selected counts, serious-event precision/recall, and captured severity
burden. Undefined recall or burden fractions are `null` when their denominator
is zero. Inspect the monthly results and slices rather than replacing them with
one multi-month Spearman coefficient. These retrospective outcomes are not the
causal benefit of performing an audit. Operational acceptance still requires an
agreed baseline, capacity, utility, uncertainty and intervention policy.

## Rehearse a reserved final period

Before any model/feature/threshold selection, reserve later complete cohorts.
The following synthetic protocol demonstrates the mechanics; published
synthetic data and already-inspected results cannot become genuinely unseen
real-world evidence by repartitioning them. Do not tune to the final report.

```bash
uv run --no-sync python -m sqm_ai.build1.evaluation reserve --input data/supplier_month.parquet --final-start 2026-03-01 --output data/build1_partition
uv run --no-sync python -m sqm_ai.build1.train --input data/build1_partition/development.parquet --output models/build1_reserved
uv run --no-sync python -m sqm_ai.build1.evaluation evaluate-final --partition data/build1_partition --models models/build1_reserved --output evaluations/build1_final
```

The partition keeps final months out of development and purges earlier rows
whose labels had not matured at the first final prediction cutoff. Both
datasets and the split record have checksums. The final evaluator verifies
partition integrity and maturity, checks that the model's training-data hash
matches the reserved development dataset, verifies artifact checksums, and
retains predictions and the monthly policy report bound to that model version.
It rejects existing output paths to preserve the earlier record. The final
model stays frozen across the evaluation period. This retrospective protocol
does not reconstruct missing source availability history.

Prespecify the decision metrics and acceptance criteria before final evaluation.
Keep a selection log, isolate final outcomes from development personnel/tools
where appropriate, and document whether any earlier experiment consumed them.
If final results influence a change, that period becomes development evidence
and another independent evaluation is needed. The command labels its output a
synthetic protocol rehearsal and leaves `release_approved` false; it does not
grant production approval or prove the history of human access to the data.
