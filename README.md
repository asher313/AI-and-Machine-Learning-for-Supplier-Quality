# sqm-ai — companion code for *Applied AI for Supplier Quality*

**A Data Scientist’s Complete Build Guide**, by Asher Nizamani.

Twenty-four chapters develop six supplier-quality systems at fictional Northlake Aerostructures. This repository contains the assembled code, instructional fragments, deterministic synthetic-data generators, behavioral tests, and run guides. The review corrects data leakage, label maturity, metrics, model/API compatibility, retrieval authorization, bounded tool execution, gateway accounting/auditing, and delivery examples while retaining the book’s progression.

The examples are teaching implementations. Synthetic data and scripted model responses demonstrate specific calculations and software behavior; they do not establish real supplier risk, machining performance, regulatory compliance, or production readiness. **The full synthetic Build 2 run fails its prediction targets** despite completing training, ONNX parity, and edge execution. Its measured results are retained rather than forced to match fictional performance tables. The separate CNC predictive gate exits nonzero, and the default scorer refuses this failed bundle; an explicit teaching-only override is documented in the Build 2 guide.

## Install the reviewed environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone this repository, and run from its root:

```bash
uv python install 3.12
uv sync --locked --extra ml
```

The committed `uv.lock` records the dependency resolution. `--locked` rejects a stale lockfile. Keep the lock with the code version; deliberate upgrades require new tests and a committed lock update. Base dependencies include sentence-transformers and its tensor dependencies, so even the initial environment is substantial. Optional groups add deep learning, evaluation, and retrieval tools:

```bash
uv sync --locked --extra ml --extra dl
# Add --extra evaluation only for Chapter 18’s optional evaluation tools.
```

On macOS, XGBoost may need OpenMP: `brew install libomp`. Hardware-specific packages such as bitsandbytes require supported platforms; the CPU CNC walkthrough does not use quantized LoRA. Optional agent frameworks use their own tested environment: [AGENTS_RUN.md](docs/AGENTS_RUN.md).

Copy `.env.example` to `.env` only when configuring services. Do not commit credentials. The numeric generators and Builds 1–4’s synthetic workflows need no provider API key, SAP account, or paid model call. Build 5 needs the optional framework dependencies; Build 6 needs a disposable PostgreSQL database.

## Generate the missing textbook datasets

```bash
uv run python scripts/generate_data.py
uv run python -m sqm_ai.verify_synthetic --data data --cnc data/cnc
```

Allow several minutes and approximately a gigabyte of free disk space. Existing output files are protected; reuse valid data or choose a new `--output` directory. For Chapters 1–11 alone, use `--skip-cnc` and omit `--cnc` from verification.

The generator constructs:

- The 29,412-row, 11-column NCR export, including category, missing-cost, supplier, and Pareto figures.
- A 36-month supplier panel with 62,518 eligible rows and 57,118 mature labeled rows; incomplete future labels remain missing.
- CNC metadata for 971,000 cycles, 931,240 inspection matches, 5,587 failures, and 51 model features; the 40,000-cycle pilot includes five-channel raw windows.
- Cobalt’s 41 before and 38 after lot-yield observations with the specified descriptive and inferential teaching results.

Generated datasets, raw arrays, trained models, and replay artifacts remain Git-ignored. Commit the generator and its verification contracts, then regenerate locally. [SYNTHETIC_DATA.md](docs/SYNTHETIC_DATA.md) defines every file, exact supported figure, chronology, and limitation. Binary file hashes can vary with serialization versions; verification also checks numerical contracts.

## Run the six Builds

| Build | Local workflow | Detailed guide |
|---|---|---|
| 1 — Supplier risk | Train on generated supplier-month rows, then score all 1,800 suppliers with the saved artifacts. | [BUILD1_RUN.md](docs/BUILD1_RUN.md) |
| 2 — Defect predictor | `uv run python -m sqm_ai.cnc.run_build2 --data data/cnc --output artifacts/build2` | [BUILD2_RUN.md](docs/BUILD2_RUN.md) |
| 3 — NCR triage | Generate triage replay data, then `uv run python -m sqm_ai.triage.run_build3`. Default shadow mode sends all decisions to review. | [BUILD3_RUN.md](docs/BUILD3_RUN.md) |
| 4 — Standards assistant | Generate the fictional corpus, index it, then `uv run python -m sqm_ai.assistant.run_build4`. No licensed AS9100 text is distributed. | [BUILD4_RUN.md](docs/BUILD4_RUN.md), [RETRIEVAL_RUN.md](docs/RETRIEVAL_RUN.md) |
| 5 — CAR drafting | In the framework environment, generate the CAR fixture, run to a persistent review pause, and resume with an explicit synthetic decision. | [BUILD5_RUN.md](docs/BUILD5_RUN.md), [AGENTS_RUN.md](docs/AGENTS_RUN.md) |
| 6 — Gateway | Run `scripts/run_build6.py` against a disposable PostgreSQL database; construct the 33/8 blocked cases and one allowed scripted response. | [BUILD6_RUN.md](docs/BUILD6_RUN.md) |

Build 1 commands:

```bash
uv run python -m sqm_ai.build1.train --input data/supplier_month.parquet --output models/build1
uv run python -m sqm_ai.build1.score_suppliers --as-of 2026-09-01 --models models/build1 --data data/supplier_month_all.parquet --output data/supplier_risk_scores.parquet
```

`--trees 30` provides a faster Build 1 execution check. Build 2 accepts `--trees 30 --epochs 2` for a shorter check. These are distinct experiments from the documented default-model runs.

## Data services, evaluation, and delivery

[docker/README.md](docker/README.md) describes a local PostgreSQL/pgvector service. Numeric file generation does not automatically load operational SQL tables. SAP/HANA extraction and production feature jobs need the documented source schemas and approved access. The schema examples and database tests show their contracts.

Live model-quality evaluation requires approved, adjudicated records, provider credentials, and explicit opt-in commands in the relevant run guide. Scripted replays are not golden-set accuracy tests. Never substitute generated response fixtures for independent quality evidence.

[CI_RUN.md](docs/CI_RUN.md) explains the root seven-stage GitLab pipeline: critical lint, offline tests, generated data, Build 1 training, an illustrative development-fold screening gate, hashed artifact registration, and staging/production **rehearsal receipts**. It does not deploy an HTTP service. [CLOUD_RUN.md](docs/CLOUD_RUN.md) covers offline AWS contract tests and dry-run request files; real cloud execution needs actual configuration and explicit launch/submit flags. [SNAPSHOT_RUN.md](docs/SNAPSHOT_RUN.md) provides an offline model-catalog check and an optional live metadata check with distinct missing/failed states.

The gateway is an injected library boundary. Applications must migrate their calls and enforce identity, credential, network, key-retention, and operational controls. It does not automatically intercept earlier direct SDK examples.

## Tests and recorded evidence

```bash
uv run pytest                  # excludes paid llm-marked tests
uv run pytest tests/test_synthetic.py
# Database tests need a disposable-server creator role:
# SQM_TEST_DATABASE_URL='postgresql+psycopg://USER@localhost/postgres' uv run pytest -m integration
```

Missing optional dependencies and database configuration produce explicit skips. Database tests create and drop UUID-named test databases; use an isolated test server. The September 15 correction run used the locked ML/DL environment and PostgreSQL: 288 passed, 12 optional-dependency skips and 5 paid-test deselections. A separate focused framework/CAR run passed 15 tests; counts overlap. The earlier September 12 environment also included the evaluation extra. Paid/nightly quality evaluations and real cloud deployments were not run. Exact validation scope and measured synthetic runs are recorded in [TECHNICAL_REVIEW_STATUS.md](docs/TECHNICAL_REVIEW_STATUS.md).

## Navigate the code

`src/sqm_ai/` contains the package; `sql/` standalone SQL; `scripts/` generators and jobs; `examples/` instructional fragments and framework comparisons; `tests/` behavioral checks; `docs/commands/` chapter command blocks. [CHAPTER_MAP.md](docs/CHAPTER_MAP.md) maps chapters to actual files. [docs/design/](docs/design/) contains the six current teaching design records.

## Errata and edition history

The September 2026 technical review and its validation boundaries are summarized in [TECHNICAL_REVIEW_STATUS.md](docs/TECHNICAL_REVIEW_STATUS.md). Git history preserves the corrections. Report an issue with the chapter/section, repository commit, environment, command, and observed result; omit company data and secrets.

## License

MIT for companion code — see [LICENSE](LICENSE). © 2026 Asher Nizamani. The manuscript has its own copyright notice.
