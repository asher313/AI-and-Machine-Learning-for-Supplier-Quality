# Chapter 22: runnable release rehearsal

The root `.gitlab-ci.yml` has seven stages and eight jobs. It generates the synthetic supplier panel, validates its contracts, trains Build 1 with 30 trees on CPU, checks the last chronological development fold, writes a release manifest, and rehearses staging/production handoff. It deploys no service and requires no model API or AWS account. Generated datasets, models and reports are not committed.

The repository is on GitHub. GitLab jobs require an imported/mirrored GitLab project and a configured `standard` runner; the YAML does not run on GitHub by itself. Mutable OS/container tags should become reviewed digests in production. The lockfile fixes Python package resolution; it does not freeze OS packages or external data.

To reproduce locally from the repository root (Python 3.12):

```bash
uv sync --locked --extra ml
uv run --no-sync python scripts/generate_data.py --output data --skip-cnc
uv run --no-sync python -m sqm_ai.verify_synthetic --data data
uv run --no-sync python -m sqm_ai.build1.train --input data/supplier_month.parquet --output artifacts/model --trees 30
uv run --no-sync python -m sqm_ai.ci.gate --metrics artifacts/model/metrics.json --min-auc 0.85 --min-recall-sev3 0.90 --report reports/evaluation.json
uv run --no-sync python -m sqm_ai.ci.pipeline register --model artifacts/model --gate reports/evaluation.json --code-sha YOUR_COMMIT --run-id UNIQUE_RUN_ID --out artifacts/registry/release.json
uv run --no-sync python -m sqm_ai.ci.pipeline deploy --release artifacts/registry/release.json --model artifacts/model --target staging --out reports/staging-rehearsal.json
```

Use a fresh model output directory for another run. Replace the revision/run placeholders with the actual code revision and a unique attempt ID; preserve uncommitted-change provenance when reviewing a working tree. Both `staging` and `production` targets only verify files and write receipts with `service_deployed: false`. Register/deploy do not execute serialized model contents. Load pickle/joblib models only from trusted sources.

The local review's 30-tree run produced final-fold ROC-AUC 0.98058 and recall 1.0 at threshold 0.04 (5,400 rows, 436 positive supplier-months). Earlier folds two and three had recall 0.856 and 0.802. These are synthetic software-demonstration results, not Northlake measurements or deployment approval. The gate checks the latest development fold and retains all fold metrics. Once these folds have informed model, feature or threshold selection, they are not an untouched final evaluation. See BUILD1_RUN.md for the separate reserved-period protocol rehearsal. Review the whole evaluation and workload before choosing real acceptance rules; do not tune repeatedly against a final holdout or lower a floor to make CI green. Build 2's synthetic performance failure remains a failure under its intended targets.

The root unit job uses existing test paths and intentionally covers a focused baseline. It is not the full ML/deep-learning/retrieval/framework validation suite. SQL tests skip unless `SQM_TEST_DATABASE_URL` points to an isolated database creator role. The integration fragment supplies disposable PostgreSQL. Optional framework testing uses `requirements-agents-lock.txt` in a separate environment; see `AGENTS_RUN.md` and `BUILD5_RUN.md`. Neural/ONNX and Ragas tests require their respective extras. Paid tests remain explicit opt-ins.

`examples/ch22_*.yml` are alternative or supplementary teaching fragments; they are not all included by the root pipeline. Cloud uploads, image scanning, downstream triggers and live evaluation require the stated operator configuration and authorization. Adding a release-critical test or security job also requires including it in the release dependency graph. A scanner report does not automatically fail a pipeline on findings. The Ragas schedule scores approved supplied answers; it does not generate answers or enforce all seven Build 4 criteria.

Validation performed locally: YAML parsed with script entries checked as strings, locked dependency installation in a fresh macOS Python 3.12 environment, focused software tests, five-fold synthetic training, gate, registration and both handoff receipts. No GitLab server CI Lint or runner execution, Linux image build, cloud registration, paid evaluation or production deployment was performed. Use the target instance's CI Lint API for rules/includes/version validation, then validate runner execution before adoption. See `docs/commands/ch22.md`.
