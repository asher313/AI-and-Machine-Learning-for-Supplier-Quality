# DESIGN RECORD — CNC inspection prediction

Owner: teaching implementation; fictional operational owners are Asher, Lena and Ravi.
Date: 2026-09-11. Status: synthetic teaching prototype; no company deployment approval.

## 1. Clarify

Evaluate whether sensor features can support inspection review under explicit latency and missed-defect costs.

## 2. Scope

The companion implements the executable learning workflow described below. No automated release, unvalidated inspection replacement or claimed real detection performance.

## 3. Architecture

Synthetic cycle summaries/raw windows → leak-safe train/calibration/test chronology → tree screening → neural second stage → ONNX parity and edge replay.

## 4. Deep dive

Feature/label joins, true sampling intervals, padding masks and training-only scaling are explicit. Define stage-one selection and combined-system denominators. A software-compatible export is distinct from a useful classifier.

## 5. Tradeoffs, risks and validation

Full synthetic two-stage training and ONNX/raw-edge parity verified. The synthetic model fails the intended detection targets; zero combined flags in the documented pilot. See BUILD2_RUN.md.

Synthetic fixtures are designed to reproduce documented descriptive contracts. They do not establish real-world performance, causal effectiveness or legal compliance. API/cloud integrations require explicit operational approval and evaluation.

## Ramp-up

Obtain representative real sensor/inspection data, validate label alignment and machine-family behavior, improve/evaluate detection under a prespecified protocol, and qualify the physical inspection fallback.

Assign a real accountable owner, dates and acceptance evidence before operational adoption. The fictional names in the narrative are not signatures.

## Changed since

2026-09-11: Recorded the reviewed implementation and verification limits. Preserve earlier decisions in version control and update the current design when the implementation changes.
