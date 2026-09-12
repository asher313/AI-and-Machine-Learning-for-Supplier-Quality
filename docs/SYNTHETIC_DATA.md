# Reproducing the textbook examples

Northlake Aerostructures, its suppliers, and these records are
fictional. No company data are included. The generator deliberately
constructs selected descriptive summaries from the book. Its model
scores are **not evidence about real suppliers or real machining**.

From the repository root, with Python 3.12 and the dependencies installed:

```bash
uv sync --extra ml
uv run python scripts/generate_data.py
uv run python -m sqm_ai.verify_synthetic --data data --cnc data/cnc
```

Allow several minutes and roughly a gigabyte of free disk space for
the full dataset. Only the 40,000-cycle pilot has stored raw windows;
the other windows are streamed through feature extraction and discarded.
No SAP connection, cloud account, model download, or API key is needed
to generate or verify these data. Run from the repository root so the
book's `data/...` paths resolve. Existing generated filenames are
protected: use a new output directory for a second run.

For Chapters 1–11 alone:

```bash
uv run python scripts/generate_data.py --skip-cnc
uv run python -m sqm_ai.verify_synthetic --data data
```

For a small CNC smoke test in a separate directory:

```bash
uv run python -m sqm_ai.cnc.synthetic --output data/cnc-small --cycles 10000 --raw-cycles 500
```

Small mode scales the failure count and pilot size. It does not satisfy
the full-size row-count contracts and cannot use the book's fixed
60,000-cycle validation blocks unchanged.

## File contracts

| File | Contents and use |
|---|---|
| `data/ncrs_2025-09_2026-08.csv` | The Chapter 1 export: 29,412 rows, 11 columns, September 2025 through August 2026. |
| `data/ncrs.parquet` | Longer synthetic NCR event history, including descriptions, owners, and recorded timestamps. |
| `data/suppliers.parquet` | 1,800 suppliers, including explicit onboarding dates. |
| `data/supplier_month_all.parquet` | 62,518 eligible supplier-months spanning 36 months. Includes the final three months with null forward labels. |
| `data/supplier_month.parquet` | 57,118 mature labeled rows used by the Part III examples; includes engineered features. |
| `data/supplier_month_features.parquet` | The same enriched, mature frame under Chapter 11's expected filename. |
| `data/cobalt_lot_fpy.csv` | 41 before and 38 after synthetic lot proportions for Chapter 3. |
| `data/cnc/cycle_features.parquet` | 931,240 matched cycles, 51 model feature columns plus serial, start time, inspection-availability time, and outcome. |
| `data/cnc/cnc_cycles.parquet` | 971,000 raw cycle metadata rows. |
| `data/cnc/cnc_inspections.parquet` | 931,240 available inspection matches, including 5,587 failures. |
| `data/cnc/cycles.parquet` | Matched cycle metadata, sorted by start time. |
| `data/cnc/pilot_cycles.parquet` | Metadata aligned with the raw pilot arrays, all from TUL-CNC-07. |
| `data/cnc/windows.npy` | 40,000 float32 pilot windows, shaped `(40000, 5, 512)`. The first 480 samples are simulated observations; the last 32 are zero padding. |
| `data/cnc/valid_lengths.npy` | Valid sample counts, to exclude padding from normalization and feature calculation. |
| `data/cnc/labels.npy` | Outcomes in raw-window order: 240 failures among 40,000 cycles. |

The `data/` directory is Git-ignored. Commit the generators, tests,
and documentation; regenerate the data locally. SHA-256 manifests
record the emitted files so accidental edits are detectable. Exact
binary hashes can vary with serialization-library versions; the
verification command also checks the actual numerical contracts.

## Which figures are reproduced exactly?

The NCR fixture matches all of these counts:

- 4,806 missing cost estimates and 2,317 open reports.
- Category counts: dimensional 13,535; cosmetic 7,062; material 6,185;
  functional 2,630. These round to the book's displayed proportions.
- Cobalt 612, Redline 498, and Apex 455 NCRs.
- 214 suppliers to cross 80% of NCRs, including the crossing supplier.
- 1,106 suppliers with at least five NCRs.
- Cobalt's August row: 47 NCRs, nine severity-3+ reports, 139 NCRs in
  June–August, FPY 0.912, OTD 0.83, audit score 71, and four open CARs.

The supplier panel matches 57,118 mature labeled rows and 5,483
positive rows (0.09599426, displayed as 0.096). The apparent arithmetic
conflict with `1800 × 33 = 59400` is resolved explicitly: 126 suppliers
join in March 2025, and one joins in November 2024. There are 2,282
excluded pre-onboarding rows in the labeled span. Consequently, the
entire 36-month panel has 62,518 eligible rows, rather than 64,800.
All 1,800 suppliers are onboarded by September 2025, so the Chapter 4
twelve-month slice still contains exactly 21,600 rows.

NCR features and both forward targets are derived from the event
history. The cutoff is the first instant of the following month;
features use events before it, and labels use `[cutoff, cutoff+90d)`.
Labels are null unless the complete forward window is covered by
the exclusive September 1, 2026 data watermark. The verifier
independently recomputes every target using cumulative sums.

The other monthly measures are constructed aggregate teaching
fixtures: they are not reconstructed from missing SAP source tables.
Delivery denominators and dates are explicit; audit and CAR measures
are simulated summaries. The Cobalt lot fixture is a separate
statistical example, not unit-level evidence supporting the monthly
FPY table or the illustrative 11,400-unit business-impact calculation.

The Cobalt fixture matches the following displayed calculations:

| Calculation | Reproduced result |
|---|---|
| Before count / mean / sample SD | 41 / 0.9502 / 0.0271 |
| After count / mean / sample SD | 38 / 0.9202 / 0.0378 |
| Welch statistic and two-sided p-value | 4.03 / 0.00015 |
| Mann–Whitney two-sided p-value | 0.00047 |
| Seed-0 percentile bootstrap difference interval | −0.0446 to −0.0161 |
| Seed-0 percentile bootstrap after-mean interval | 0.9083 to 0.9320 |

These values were deliberately moment-matched and their fixed row
ordering chosen to match the printed Monte Carlo intervals. This is
appropriate for reconstructing a fictional teaching example; it
would be inappropriate for analyzing empirical research data. Do not
interpret agreement with the book as independent statistical evidence.

The CNC generator matches the full-cycle, unmatched-cycle, failure,
and pilot counts. Its 51 feature columns are calculated from simulated
sensor windows using the same definitions as `cycle_features`. The
original manuscript's claim of 96 features was inconsistent with the
listing and is corrected. Pilot collection spans months, not two days.
All families have 48-second cycles in this fixture; reproducing the
fictional rib-web duration problem requires a separate scenario.

## Model experiments and external services

Synthetic data do not guarantee the book's AUC/AP tables, SHAP values,
threshold results, cluster memberships, drift figures, or runtimes.
Fit the models, evaluate held-out periods, and report what actually
happens. Never insert labels, future outcomes, or metadata into the
predictors to force agreement. A suitable CNC predictor frame is:

```python
from sqm_ai.cnc.features import frame
import pandas as pd

df = pd.read_parquet("data/cnc/cycle_features.parquet")
y = df["failed"].to_numpy()
X = frame(df.drop(columns=[
    "failed", "serial", "cycle_start", "inspection_completed_at"
]).to_dict("records"))
```

The data make local statistical and model examples executable.
Database queries still need the relevant schemas and loaded tables;
SAP extraction needs an authorized HANA system. The language-model,
retrieval, and deployment Builds also require their own documents,
service configurations, and credentials. Synthetic numeric datasets
alone do not turn those external integrations into offline programs.
