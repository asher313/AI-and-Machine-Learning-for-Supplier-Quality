# sqm-ai — companion code for *Applied AI for Supplier Quality*

**Applied AI for Supplier Quality — A Data Scientist's Complete
Build Guide**, by Asher Nizamani.

This repository holds every code listing in the book, in the
layout the book builds toward. It follows one engineer, Asher,
through twenty-four chapters at Northlake Aerostructures — a
fictional aerostructures manufacturer in Tulsa — as he turns a
trailing-twelve-month nonconformance export into six production
systems: a supplier risk score, a defect predictor that runs at
the machine, an NCR triage classifier, an AS9100 assistant
grounded in the company's own documents, a corrective-action
drafting crew, and a compliance gateway that every language-model
call in the estate passes through. The chapters build each layer
of the Supplier Intelligence Stack in order, and every file here
is the artifact one of those chapters produced. The repository contains runnable components and instructional
listings. Some demonstrations require services or reader-supplied
inputs; the setup and reproducibility notes identify those boundaries.

## The Supplier Intelligence Stack

```text
 ┌──────────────────────────────────────────────────────┐
 │ 7  DELIVERY     GitLab CI/CD · AWS                   │
 ├──────────────────────────────────────────────────────┤
 │ 6  TRUST        tests · evals · guardrails · audit   │
 ├──────────────────────────────────────────────────────┤
 │ 5  AGENTS       tool loop · LangGraph · CrewAI · …   │
 ├──────────────────────────────────────────────────────┤
 │ 4  KNOWLEDGE    chunks · embeddings · vectors · RAG  │
 ├──────────────────────────────────────────────────────┤
 │ 3  LANGUAGE     LLM calls · structured outputs       │
 ├──────────────────────────────────────────────────────┤
 │ 2  LEARNING     scikit-learn · XGBoost · PyTorch     │
 ├──────────────────────────────────────────────────────┤
 │ 1  DATA         SQL · SAP HANA · pandas              │
 ├──────────────────────────────────────────────────────┤
 │ 0  MATH         linear algebra · probability · stats │
 └──────────────────────────────────────────────────────┘
```

## Layout

```text
src/sqm_ai/      the package the book builds, chapter by chapter
  build1/        Build 1 — supplier risk score (Ch 10)
  cnc/           Build 2 — defect prediction from sensors (Ch 14)
  triage/        Build 3 — NCR triage classifier (Ch 16)
  retrieval/     Build 4 — retrieval (Ch 17)
  assistant/     Build 4 — the AS9100 assistant (Ch 18)
  agent/         Build 5 — tools and the no-framework loop (Ch 19)
  car/           Build 5 — the CAR drafting crew (Ch 20)
  gateway/       Build 6 — the compliance gateway (Ch 21)
  dl/            the PyTorch package (Ch 11-13)
  ci/            the CI quality gate (Ch 22)
  aws/           S3, Bedrock, CloudWatch, Secrets Manager (Ch 23)
sql/             standalone SQL the chapters name by path
scripts/         one-shot jobs the chapters name by path
tests/           the book's tests, plus tests/test_imports.py
examples/        illustrative listings with no file path of their
                 own: the algorithm catalogue, the math demos,
                 the framework comparisons, Appendix D's reprints
docs/commands/   every shell and PowerShell block, by chapter
docs/CHAPTER_MAP.md   this map, plus the section behind each file
docker/          Postgres + pgvector for a laptop
k8s/, lambdas/   Chapter 23's deployment manifests
```

## Setup

The book uses **uv** for interpreters, environments, packages,
and the lockfile, and pins **Python 3.12**.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.12
uv sync
```

```powershell
# Windows PowerShell
winget install --id astral-sh.uv -e
uv python install 3.12
uv sync
```

`uv sync` installs the runtime dependencies from
`pyproject.toml` and writes `uv.lock`. Parts III and IV add
larger libraries; install them when you reach them:

```bash
uv sync --extra ml      # scikit-learn, XGBoost, LightGBM, SHAP, Optuna
uv sync --extra dl      # PyTorch, transformers, PEFT, ONNX
```

Then copy the settings template and fill it in:

```bash
cp .env.example .env
```

`.env` is never committed. `Settings` in
`src/sqm_ai/settings.py` reads it once, and `get_settings()` is
the only approved way to reach it. In CI the same variables come
from GitLab protected variables (Chapter 22); in the restricted
enclave they come from AWS Secrets Manager (Chapter 23).

### Synthetic teaching data

Generate the textbook datasets locally; generated data stay out of Git:

```bash
uv run python scripts/generate_data.py
uv run python -m sqm_ai.verify_synthetic --data data --cnc data/cnc
```

See [the reproducibility guide](docs/SYNTHETIC_DATA.md) for file contracts,
exact descriptive figures, smaller runs, and the distinction between
constructed examples and measured model performance.

### The database

Chapters 4 onward assume a PostgreSQL analytics copy, and
Chapters 17-18 assume the **pgvector** extension on it. There is
a compose file for a laptop:

```bash
docker compose -f docker/docker-compose.yml up -d
psql "postgresql://sqm:sqm@localhost:5432/sqm_analytics" \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

Point `DATABASE_URL` at it, then create the tables:
`src/sqm_ai/sql/supplier_month.sql` (Ch 4),
`sql/doc_chunks.sql` and `sql/doc_parents.sql` (Ch 17),
`src/sqm_ai/triage/schema.sql` (Ch 16),
`src/sqm_ai/gateway/schema.sql` (Ch 21).
See `docker/README.md`.

### Tests

```bash
uv run pytest              # llm-marked tests are deselected
uv run pytest -m llm       # runs against the real model, on purpose
uv run pytest -m integration   # needs a live database
```

## Running the Builds

| Build | Chapter | Command |
|---|---|---|
| **1 — Supplier risk score** | 10 | `uv run python -m sqm_ai.features` to rebuild `sqm.supplier_month`, then `uv run python -m sqm_ai.build1.train` and `uv run python -m sqm_ai.build1.score_suppliers` (the nightly 03:00 job). `src/sqm_ai/build1/sql/features.sql` and `labels.sql` build the modelling frame. |
| **2 — Defect predictor** | 14 | `uv run python -m sqm_ai.cnc.train_stage1`, then `train_stage2`, then `export` (which checks ONNX parity to `1e-5` against the PyTorch model). `sqm_ai/cnc/edge.py` is what runs at the cell. |
| **3 — NCR triage classifier** | 16 | `sqm_ai/triage/schema.sql` first, then the pipeline: `rules` → `classify` → `guardrails` → `trace` → the review queue. Tests: `uv run pytest tests/triage`. |
| **4 — AS9100 assistant** | 17-18 | `uv run python scripts/index_corpus.py` to build the index, then `sqm_ai.assistant.answer`. `uv run python scripts/eval_generation.py` runs the weekly evaluation. |
| **5 — CAR drafting crew** | 19-20 | `sqm_ai.agent.loop` is the no-framework baseline; `sqm_ai.car.graph` is the shipped LangGraph version. `examples/agents/` holds the LangGraph, AutoGen, and CrewAI comparisons from Chapter 19. |
| **6 — Compliance gateway** | 21 | A library, not a proxy service: `from sqm_ai.gateway import complete`. `sqm_ai/gateway/schema.sql` creates the audit table. |

Supporting jobs: `uv run python -m sqm_ai.extract` (the nightly
HANA extract, Ch 5), `uv run python -m sqm_ai.anomalies` (the
weekly anomaly report, Ch 6), `uv run python scripts/train_baseline.py`
(Chapter 7's baseline, the floor every later model must clear),
`uv run python scripts/verify_snapshot.py` (Ch 24 — run it twice
a year and before any design review).

`.gitlab-ci.yml` is Chapter 22's Build 1 pipeline: lint · test ·
data · train · evaluate · register · deploy, with the leakage
test in the `test` stage's integration job.

## Chapter → file map

| Chapter | Files produced |
|---|---|
| **Chapter 1** — Asher's First Week: Supplier Quality in Numbers | `examples/ch01_asher_at_northlake.py` · `examples/ch01_asher_at_northlake_2.py` |
| **Chapter 2** — The Data Scientist's Workbench | `.env.example` · `.gitignore` · `.python-version` · `Makefile` · `docs/commands/ch02.md` · `examples/ch02_comprehensions.py` · `examples/ch02_context_managers.py` · `examples/ch02_dataclass_or_pydantic_the_boundary_rule.py` · `examples/ch02_exception_classes.py` · `examples/ch02_generators.py` · `examples/ch02_logging_and_observability_from_day_one.py` · `examples/ch02_minimal_gitlab_ci.yml` · `examples/ch02_type_hints.py` · `pyproject.toml` · `src/sqm_ai/errors.py` · `src/sqm_ai/log.py` · `src/sqm_ai/ncr.py` · `src/sqm_ai/settings.py` · `src/sqm_ai/timing.py` · `src/sqm_ai/triage/__init__.py` · `tests/test_features.py` · `tests/test_ncr.py` · `tests/test_triage.py` |
| **Chapter 3** — The Math You Will Actually Use | `docs/commands/ch03.md` · `examples/ch03_asher_at_northlake.py` · `examples/ch03_asher_at_northlake_2.py` · `examples/ch03_distributions_you_will_meet.py` · `examples/ch03_distributions_you_will_meet_2.py` · `examples/ch03_eigenvalues_svd_and_low_rank.py` · `examples/ch03_gradients_and_the_chain_rule.py` · `examples/ch03_information_theory_in_three_formulas.py` · `examples/ch03_information_theory_in_three_formulas_2.py` · `examples/ch03_matrices_every_layer_is_a_multiply.py` · `examples/ch03_probability_bayes_and_the_defect_detector_tr.py` · `examples/ch03_statistics_for_the_quality_office.py` · `examples/ch03_vectors_dot_products_and_why_cosine_similari.py` · `examples/ch03_vectors_dot_products_and_why_cosine_similari_2.py` · `examples/ch03_vectors_dot_products_and_why_cosine_similari_3.py` · `examples/ch03_vectors_dot_products_and_why_cosine_similari_4.py` · `src/sqm_ai/stats.py` |
| **Chapter 4** — SQL for Supplier Data | `docs/commands/ch04.md` · `examples/ch04_ctes_and_recursion.sql` · `examples/ch04_ctes_and_recursion_2.sql` · `examples/ch04_joins_illustrated_with_suppliers_and_ncrs.sql` · `examples/ch04_joins_illustrated_with_suppliers_and_ncrs_2.sql` · `examples/ch04_joins_illustrated_with_suppliers_and_ncrs_3.sql` · `examples/ch04_joins_illustrated_with_suppliers_and_ncrs_4.sql` · `examples/ch04_reading_an_explain_plan.sql` · `examples/ch04_reading_an_explain_plan_2.sql` · `examples/ch04_select_filter_aggregate.sql` · `examples/ch04_select_filter_aggregate_2.sql` · `examples/ch04_subqueries_exists_case.sql` · `examples/ch04_subqueries_exists_case_2.sql` · `examples/ch04_subqueries_exists_case_3.sql` · `examples/ch04_subqueries_exists_case_4.sql` · `examples/ch04_subqueries_exists_case_5.sql` · `examples/ch04_the_sqm_query_cookbook.sql` · `examples/ch04_the_sqm_query_cookbook_2.sql` · `examples/ch04_the_sqm_query_cookbook_3.sql` · `examples/ch04_window_functions.sql` · `examples/ch04_window_functions_2.sql` · `examples/ch04_window_functions_3.sql` · `examples/ch04_window_functions_4.sql` · `src/sqm_ai/features.py` · `src/sqm_ai/sql/supplier_month.sql` · `tests/test_supplier_month.py` |
| **Chapter 5** — SAP HANA and the Enterprise Data Estate | `docs/commands/ch05.md` · `examples/ch05_calculation_views.sql` · `examples/ch05_dialect_differences_that_bite.sql` · `examples/ch05_dialect_differences_that_bite_2.sql` · `examples/ch05_python_to_hana.py` · `examples/ch05_python_to_hana_2.py` · `examples/ch05_training_where_the_data_lives.py` · `src/sqm_ai/extract.py` · `src/sqm_ai/hana.py` · `src/sqm_ai/sql/extract_manifest.yaml` · `tests/test_extract_contract.py` |
| **Chapter 6** — NumPy and pandas: Shaping Supplier Data | `docs/commands/ch06.md` · `examples/ch06_arrays_shapes_axes.py` · `examples/ch06_arrays_shapes_axes_2.py` · `examples/ch06_arrays_shapes_axes_3.py` · `examples/ch06_arrays_shapes_axes_4.py` · `examples/ch06_broadcasting_the_one_rule.py` · `examples/ch06_broadcasting_the_one_rule_2.py` · `examples/ch06_dataframes_in_and_out.py` · `examples/ch06_dataframes_in_and_out_2.py` · `examples/ch06_groupby_the_sqm_workhorse.py` · `examples/ch06_groupby_the_sqm_workhorse_2.py` · `examples/ch06_groupby_the_sqm_workhorse_3.py` · `examples/ch06_groupby_the_sqm_workhorse_4.py` · `examples/ch06_indexing_masks_and_matrix_ops.py` · `examples/ch06_indexing_masks_and_matrix_ops_2.py` · `examples/ch06_indexing_masks_and_matrix_ops_3.py` · `examples/ch06_indexing_masks_and_matrix_ops_4.py` · `examples/ch06_indexing_masks_and_matrix_ops_5.py` · `examples/ch06_inspect_first_always.py` · `examples/ch06_inspect_first_always_2.py` · `examples/ch06_merge_concat_pivot.py` · `examples/ch06_merge_concat_pivot_2.py` · `examples/ch06_merge_concat_pivot_3.py` · `examples/ch06_selecting_and_filtering.py` · `examples/ch06_selecting_and_filtering_2.py` · `examples/ch06_the_sqm_pandas_cookbook.py` · `examples/ch06_the_sqm_pandas_cookbook_2.py` · `examples/ch06_the_sqm_pandas_cookbook_3.py` · `examples/ch06_the_sqm_pandas_cookbook_4.py` · `examples/ch06_the_sqm_pandas_cookbook_5.py` · `examples/ch06_time_series.py` · `examples/ch06_time_series_2.py` · `examples/ch06_time_series_3.py` · `examples/ch06_time_series_4.py` · `examples/ch06_transforming.py` · `examples/ch06_transforming_2.py` · `examples/ch06_transforming_3.py` · `examples/ch06_transforming_4.py` · `examples/ch06_transforming_5.py` · `src/sqm_ai/anomalies.py` · `tests/test_anomalies.py` |
| **Chapter 7** — The scikit-learn Workflow, End to End | `examples/ch07_asher_at_northlake.py` · `examples/ch07_cross_validation_done_right.py` · `examples/ch07_cross_validation_done_right_2.py` · `examples/ch07_encoders.py` · `examples/ch07_encoders_2.py` · `examples/ch07_imputers.py` · `examples/ch07_saving_and_loading.py` · `examples/ch07_scalers.py` · `examples/ch07_scalers_2.py` · `examples/ch07_the_eleven_step_template.py` · `examples/ch07_the_five_methods_every_estimator_has.py` · `examples/ch07_the_five_methods_every_estimator_has_2.py` · `scripts/train_baseline.py` · `src/sqm_ai/features.py` |
| **Chapter 8** — The Algorithm Catalog: When and Why | `examples/ch08_decision_trees.py` · `examples/ch08_gradient_boosting_xgboost_and_lightgbm.py` · `examples/ch08_gradient_boosting_xgboost_and_lightgbm_2.py` · `examples/ch08_linear_and_logistic_regression.py` · `examples/ch08_linear_and_logistic_regression_2.py` · `examples/ch08_naive_bayes.py` · `examples/ch08_random_forest.py` · `examples/ch08_svms_and_k_nn.py` · `examples/ch08_svms_and_k_nn_2.py` · `examples/ch08_unsupervised_k_means_dbscan_pca.py` · `examples/ch08_unsupervised_k_means_dbscan_pca_2.py` · `examples/ch08_unsupervised_k_means_dbscan_pca_3.py` · `src/sqm_ai/bakeoff.py` |
| **Chapter 9** — Imbalance, Metrics, Tuning, and Explainability | `examples/ch09_asher_at_northlake.py` · `examples/ch09_asher_at_northlake_2.py` · `examples/ch09_class_weights_resampling_and_threshold_tunin.py` · `examples/ch09_class_weights_resampling_and_threshold_tunin_2.py` · `examples/ch09_class_weights_resampling_and_threshold_tunin_3.py` · `examples/ch09_explainability.py` · `examples/ch09_explainability_2.py` · `examples/ch09_explainability_3.py` · `examples/ch09_explainability_4.py` · `examples/ch09_feature_engineering_patterns.py` · `examples/ch09_hyperparameter_search.py` · `examples/ch09_hyperparameter_search_2.py` · `examples/ch09_metrics_decoded.py` |
| **Chapter 10** — Build 1: The Supplier Risk Score | `src/sqm_ai/build1/explain.py` · `src/sqm_ai/build1/features.py` · `src/sqm_ai/build1/score_suppliers.py` · `src/sqm_ai/build1/sql/features.sql` · `src/sqm_ai/build1/sql/labels.sql` · `src/sqm_ai/build1/train.py` · `tests/build1/test_explain_sums_to_prediction.py` · `tests/build1/test_features_ignore_the_future.py` |
| **Chapter 11** — PyTorch From Tensors to the Training Loop | `examples/ch11_autograd.py` · `examples/ch11_autograd_2.py` · `examples/ch11_autograd_3.py` · `examples/ch11_devices.py` · `examples/ch11_devices_2.py` · `examples/ch11_optimizers_schedulers_losses.py` · `examples/ch11_optimizers_schedulers_losses_2.py` · `examples/ch11_optimizers_schedulers_losses_3.py` · `examples/ch11_saving_loading_checkpointing_exporting.py` · `examples/ch11_saving_loading_checkpointing_exporting_2.py` · `examples/ch11_saving_loading_checkpointing_exporting_3.py` · `examples/ch11_tensors.py` · `examples/ch11_tensors_2.py` · `src/sqm_ai/dl/data.py` · `src/sqm_ai/dl/models.py` · `src/sqm_ai/dl/risk_mlp.py` · `src/sqm_ai/dl/train.py` |
| **Chapter 12** — Architectures From Scratch: MLP, CNN, RNN, Transformer | `examples/ch12_a_basic_convolutional_network.py` · `examples/ch12_multi_head_attention.py` · `examples/ch12_position_wise_feed_forward.py` · `examples/ch12_positional_encoding.py` · `examples/ch12_recurrent_networks_rnn_and_lstm.py` · `examples/ch12_recurrent_networks_rnn_and_lstm_2.py` · `examples/ch12_scaled_dot_product_attention.py` · `examples/ch12_the_block_and_the_model.py` · `examples/ch12_the_block_and_the_model_2.py` · `examples/ch12_the_causal_mask.py` · `examples/ch12_the_multi_layer_perceptron.py` · `examples/ch12_the_residual_block_and_the_key_line.py` · `src/sqm_ai/dl/cnc_cnn.py` |
| **Chapter 13** — Transfer Learning, HuggingFace, LoRA, and the TensorFlow Map | `examples/ch13_distributed_training_conceptually.py` · `examples/ch13_huggingface_transformers.py` · `examples/ch13_huggingface_transformers_2.py` · `examples/ch13_lora_and_qlora.py` · `examples/ch13_lora_and_qlora_2.py` · `examples/ch13_lora_and_qlora_3.py` · `examples/ch13_tensorflow_and_keras_the_map.py` · `examples/ch13_tensorflow_and_keras_the_map_2.py` · `examples/ch13_tensorflow_and_keras_the_map_3.py` · `examples/ch13_tensorflow_and_keras_the_map_4.py` · `examples/ch13_transfer_learning.py` · `examples/ch13_transfer_learning_2.py` · `src/sqm_ai/dl/ncr_finetune.py` · `src/sqm_ai/dl/ncr_zero_shot.py` |
| **Chapter 14** — Build 2: Defect Prediction From Machine Sensors | `src/sqm_ai/cnc/edge.py` · `src/sqm_ai/cnc/export.py` · `src/sqm_ai/cnc/features.py` · `src/sqm_ai/cnc/train_stage1.py` · `src/sqm_ai/cnc/train_stage2.py` |
| **Chapter 15** — Calling a Language Model in Production | `docs/commands/ch15.md` · `examples/ch15_async_and_bounded_concurrency.py` · `examples/ch15_local_inference.py` · `examples/ch15_local_inference_2.py` · `examples/ch15_model_cascades.py` · `examples/ch15_prompt_caching.py` · `examples/ch15_retries_with_backoff.py` · `examples/ch15_semantic_caching.py` · `examples/ch15_streaming_and_multi_turn.py` · `examples/ch15_streaming_and_multi_turn_2.py` · `examples/ch15_structured_outputs.py` · `examples/ch15_structured_outputs_2.py` · `examples/ch15_structured_outputs_3.py` · `examples/ch15_the_canonical_call.py` · `src/sqm_ai/llm.py` |
| **Chapter 16** — Build 3: The NCR Triage Classifier | `examples/ch16_shipping_it_shadow_mode.sql` · `examples/ch16_the_schema_and_the_review_queue.sql` · `examples/ch16_tracing_every_call.py` · `src/sqm_ai/triage/classify.py` · `src/sqm_ai/triage/guardrails.py` · `src/sqm_ai/triage/review.py` · `src/sqm_ai/triage/rules.py` · `src/sqm_ai/triage/schema.py` · `src/sqm_ai/triage/schema.sql` · `src/sqm_ai/triage/trace.py` · `tests/data/golden_ncrs.json` · `tests/triage/test_golden.py` · `tests/triage/test_pipeline.py` · `tests/triage/test_structure.py` |
| **Chapter 17** — Retrieval: Grounding the Model in Your Documents | `docs/commands/ch17.md` · `examples/ch17_advanced_patterns.py` · `examples/ch17_advanced_patterns_2.py` · `examples/ch17_chunking.py` · `examples/ch17_chunking_2.py` · `examples/ch17_chunking_3.py` · `examples/ch17_embeddings.py` · `examples/ch17_the_minimal_pipeline.py` · `examples/ch17_vector_stores.py` · `scripts/index_corpus.py` · `sql/chunk_search.sql` · `sql/doc_chunks.sql` · `sql/doc_parents.sql` · `src/sqm_ai/retrieval/embed.py` · `src/sqm_ai/retrieval/expand.py` · `src/sqm_ai/retrieval/fuse.py` · `src/sqm_ai/retrieval/rerank.py` · `src/sqm_ai/retrieval/store.py` |
| **Chapter 18** — Build 4: The AS9100 Assistant | `scripts/eval_generation.py` · `src/sqm_ai/assistant/access.py` · `src/sqm_ai/assistant/answer.py` · `src/sqm_ai/assistant/judge.py` · `src/sqm_ai/assistant/metrics.py` · `src/sqm_ai/assistant/prompts.py` · `src/sqm_ai/assistant/retrieve.py` · `src/sqm_ai/assistant/validate.py` · `src/sqm_ai/assistant/versioning.py` · `tests/assistant/test_access.py` · `tests/sql/no_double_open.sql` |
| **Chapter 19** — Agents and Frameworks: Plain Python, LangGraph, AutoGen, CrewAI | `examples/agents/autogen_ncr.py` · `examples/agents/crewai_ncr.py` · `examples/agents/langgraph_ncr.py` · `src/sqm_ai/agent/loop.py` · `src/sqm_ai/agent/tools.py` |
| **Chapter 20** — Build 5: The CAR Drafting Crew | `examples/ch20_measuring_a_draft.sql` · `src/sqm_ai/car/graph.py` · `src/sqm_ai/car/measure.py` · `src/sqm_ai/car/memory.py` · `src/sqm_ai/car/nodes.py` · `src/sqm_ai/car/redact.py` · `src/sqm_ai/car/root_cause.py` · `src/sqm_ai/car/state.py` · `src/sqm_ai/car/validate.py` |
| **Chapter 21** — Build 6: The Compliance Gateway | `examples/ch21_asher_at_northlake.sql` · `src/sqm_ai/gateway/__init__.py` · `src/sqm_ai/gateway/audit.py` · `src/sqm_ai/gateway/detectors.py` · `src/sqm_ai/gateway/guardrails.py` · `src/sqm_ai/gateway/policy.py` · `src/sqm_ai/gateway/router.py` · `src/sqm_ai/gateway/schema.sql` · `tests/gateway/test_pre_check.py` |
| **Chapter 22** — GitLab CI/CD for Machine Learning | `.gitlab-ci.yml` · `docs/commands/ch22.md` · `examples/ch22_artifacts_and_model_storage.yml` · `examples/ch22_keywords_variables_secrets_rules.yml` · `examples/ch22_keywords_variables_secrets_rules_2.yml` · `examples/ch22_ml_specific_test_jobs.yml` · `examples/ch22_ml_specific_test_jobs_2.yml` · `examples/ch22_multi_project_pipelines.yml` · `examples/ch22_runners_and_gpus.yml` · `examples/ch22_security_scanning.yml` · `examples/ch22_security_scanning_2.yml` · `examples/ch22_the_mental_model.yml` · `src/sqm_ai/ci/gate.py` |
| **Chapter 23** — AWS for the ML Engineer | `docs/commands/ch23.md` · `examples/ch23_bedrock_the_enclave_path.py` · `examples/ch23_ecs_task_definition.json` · `examples/ch23_iam_least_privilege_policy.json` · `k8s/ncr-classifier.yaml` · `lambdas/ncr_received/handler.py` · `scripts/sm_train.py` · `scripts/spot_train.py` · `src/sqm_ai/aws/enclave.py` · `src/sqm_ai/aws/metrics.py` · `src/sqm_ai/aws/secrets.py` · `src/sqm_ai/aws/storage.py` |
| **Chapter 24** — Presenting the Stack: The Architecture Review | `scripts/verify_snapshot.py` |
| **Appendix D** — Code You Should Be Able to Write From Memory | `examples/appendixD_agent_loop.py` · `examples/appendixD_minimal_rag.py` · `examples/appendixD_retry_helper.py` · `examples/appendixD_sklearn_template.py` · `examples/appendixD_structured_output.py` · `examples/appendixD_training_loop.py` · `examples/appendixD_window_query.sql` |

`docs/CHAPTER_MAP.md` repeats this map with the section behind
every file.

## Errata

None recorded yet.

| Chapter | Page / section | Correction |
|---|---|---|

## License

MIT — see `LICENSE`. © 2026 Asher Nizamani.
