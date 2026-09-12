# September 2026 technical review — validation record

All 24 chapters have received technical reading and correction passes. Cross-book reconciliation and final manuscript production are separate from software validation; this file does not certify publication layout or deployed operation.

## Verified execution

- Four requested numeric fixture families, with count, chronology, feature, and statistical contracts; generated data remain outside Git.
- Default five-fold Build 1 training and 1,800-supplier scoring; a separate five-fold, 20-epoch MLP comparison. See [synthetic_build1_mlp_run.json](synthetic_build1_mlp_run.json).
- Full default Build 2 training (500 trees, 30 epochs), ONNX parity, and raw-window edge checks. Its test caught 0 of 32 failures and raised 0 holds; precision is undefined. **Prediction targets failed.** See [synthetic_build2_run.json](synthetic_build2_run.json).
- PyTorch architecture, masks, training, serialization, and small offline transfer-learning trainer contracts.
- Build 3’s 20-record shadow replay; Build 4’s 13-document/25-chunk scoped lexical replay; Build 5’s persistent review pause/resume; Build 6’s constructed 33/8 blocks and one allowed local response. All make zero generation API calls.
- Real PostgreSQL 16.15/pgvector checks in isolated UUID-named databases: source extraction, date/label rules, dense/sparse/final/parent authorization, atomic supersession, scoped histories, mature CAR recurrence, gateway audit and concurrent reservations.
- All four actual orchestration runtimes using one guarded scripted kernel, including persisted LangGraph resume. The optional framework environment is frozen in `requirements-agents-lock.txt`.
- AWS contract checks with Botocore Stubber and service request-shape validation. No AWS resources created. The training container was not built; an external HTTP application is still needed for the serving-manifest examples.
- Fresh `uv sync --locked --extra ml` installation on macOS. The 30-tree CI rehearsal trained five chronological folds and registered seven verified artifacts; its last fold passed the illustrative gate. Earlier folds did not all pass. No GitLab server pipeline or Linux-container execution is claimed.
- Real Anthropic SDK pagination through a two-page HTTP mock; no live model availability claim.

## Test counts

The final September 12, 2026 run with the isolated PostgreSQL server recorded **247 passed, 7 skipped, 5 deselected**. The earlier offline run without PostgreSQL recorded **230 passed, 23 skipped, 5 deselected** before the final pagination test was added. The seven remaining skips concern optional framework dependencies; the separate framework environment passed **14 tests**, including CAR persistence and the PostgreSQL history contract. These suites overlap and must not be added into a unique-test total. The final 247-test run includes Chapter 24’s two-page SDK pagination test. The final optional-framework rerun passed 14 tests. Final focused reconciliation checks are recorded in the review deliverable.

Critical Ruff checks (`E9,F63,F7,F82`) passed for package, tests, scripts, and Lambda sources. This is not a claim that all formatting/style rules pass for instructional fragments.

## Limits of this evidence

The tables describing Northlake business outcomes, latency, costs, and model comparisons are fictional unless explicitly identified as measured synthetic runs. Numeric fixture construction cannot establish causal effects, real defect observability, deployment calibration, or fairness. Scripted responses establish software contracts, not live answer quality or human approval effectiveness.

No paid generation evaluation, production SAP access, licensed standards corpus, large pretrained-model quality experiment, cloud deployment, or operational user acceptance was performed. Production identity/UI integration, monitoring, incident procedures, authorized data handling, retained encryption keys, and genuine domain validation remain organization-specific work described in the chapters and design records.
