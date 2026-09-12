# Technical-review checkpoint

The `technical-review-2026-09` branch is an ongoing textbook/companion review.
The complete manuscript and KDP layout are not yet finalized. Chapters 1–16
have received technical reading and correction passes; later chapters and
cross-book metadata still require review. This is not a claim that every
external-service Build or the entire repository test suite passes.

Verified local work includes:

- All four requested synthetic numeric fixtures, with independent count,
  statistical, chronology, and feature checks.
- Default five-fold Build 1 training and scoring of all 1,800 suppliers.
- Five-fold, 20-epoch MLP comparison with independent checkpoint selection.
- PyTorch training, architecture, causal-mask, and serialization checks.
- Tiny offline category and severity Trainer workflows and response validation.
- Full default synthetic Build 2 walkthrough, ONNX parity, and edge checks.
  Its predictive performance is poor; see BUILD2_RUN.md.
- Nine PostgreSQL integration checks on isolated disposable test databases.
- Chapter 15 response/retry/cache boundaries and Chapter 16 triage composition,
  with generated offline replay fixtures and explicit opt-in live evaluation.

Generated datasets and model artifacts remain outside Git. The numeric generator
and verified teaching workflows do not replace SAP, source documents, cloud
services, deployment configurations, or approval processes needed by later Builds.
Paid model calls and large pretrained-model training have not been run in this
review. Actual synthetic experiment records are stored separately from fictional
manuscript performance tables.
