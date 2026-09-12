# DESIGN RECORD — Supplier risk scoring

Owner: teaching implementation; fictional operational owners are Asher, Lena and Ravi.
Date: 2026-09-11. Status: synthetic teaching prototype; no company deployment approval.

## 1. Clarify

Batch ranking and review support for 1,800 fictional suppliers. Predict mature 90-day outcomes from prior feature snapshots.

## 2. Scope

The companion implements the executable learning workflow described below. No supplier suspension or automatic quality disposition. Production utility on real data is not established.

## 3. Architecture

Synthetic NCR and supplier-month generator → maturity/feature validation → chronological XGBoost pipelines → calibrated probability and harm predictions → ranked report with SHAP drivers.

## 4. Deep dive

Preprocessing is fit within each training partition; calibration uses an inner chronological tail with mature labels. No future-label columns enter predictors. Explain raw harm predictions and identify any display floor separately.

## 5. Tradeoffs, risks and validation

Full synthetic training and 1,800-score generation verified. The 30-tree CI rehearsal gates the final fold and preserves earlier fold recall failures. See BUILD1_RUN.md and CI_RUN.md.

Synthetic fixtures are designed to reproduce documented descriptive contracts. They do not establish real-world performance, causal effectiveness or legal compliance. API/cloud integrations require explicit operational approval and evaluation.

## Ramp-up

Validate real definitions, data history, outcome maturity, workload and calibration; compare an incumbent baseline; enforce serving/data authorization and monitor delayed outcomes.

Assign a real accountable owner, dates and acceptance evidence before operational adoption. The fictional names in the narrative are not signatures.

## Changed since

2026-09-11: Recorded the reviewed implementation and verification limits. Preserve earlier decisions in version control and update the current design when the implementation changes.
