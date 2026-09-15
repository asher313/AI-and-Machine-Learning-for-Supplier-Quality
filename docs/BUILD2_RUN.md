# Running the synthetic Build 2 walkthrough

From the repository root:

```bash
uv sync --extra ml --extra dl
uv run python scripts/generate_data.py
uv run python -m sqm_ai.cnc.run_build2 --data data/cnc --output artifacts/build2
```

If the data already exist, skip generation. Use another output directory for
another training run. `--trees 30 --epochs 2` provides a quicker execution check.
On macOS, XGBoost may require `brew install libomp`.

The walkthrough fits stage 1 on early all-cell feature history, chooses its gate
on a later calibration period, and freezes it before developing stage 2 on
later pilot windows. Subsequent pilot periods select the CNN checkpoint and
threshold, followed by an untouched 6,000-cycle test. Label availability is
checked before each later development period. Only the one-cell pilot has
stored raw windows, so this is not an all-cell validation of the cascade.

The bundle contains numeric categorical preprocessing, channel normalization,
unrounded thresholds, ONNX models, a model version, and file hashes. Stage-1
and stage-2 exports are checked against native predictions. Raw-window edge
checks cover both gated and ungated examples. Load only trusted bundles:
joblib deserialization is not made safe merely by a matching checksum.

For valid raw windows and known cycle-start context:

```python
from sqm_ai.cnc.edge import CycleScorer
scorer = CycleScorer("artifacts/build2", allow_failed_demo=True)
# decision = scorer.score(window, context)
```

`no_model_flag` is not an inspection pass or release authorization. `qa_hold`
is a suggestion for the approved human quality process. The example does not
connect to a machine interlock, HMI, production database, or notification system.
It rejects unsupported long windows instead of silently truncating them.

## What the validation establishes

The full default run used 500 trees and 30 CNN epochs. It completed training,
export parity, and raw-window edge checks. Its final test caught **0 of 32
failures** and raised **0 holds**; precision is undefined, not zero. Stage 1
had AP 0.00756 and gate recall 0.0625 on the pilot test. These poor results are
recorded in [synthetic_build2_run.json](synthetic_build2_run.json).

This validates software execution, not predictive usefulness. It does not meet
the textbook's fictional quality floors and must not be described as a model
ready for operational use. The generator reproduces descriptive counts, not
the fictional AP, recall, precision, thresholds, or runtimes. The policy was
selected on development data; it was not retuned to make the test look better.

The 10 Hz traces demonstrate software handling of low-rate signals. They do
not establish that audible chatter or any real machining defect is observable
at that rate. Sensor semantics and bandwidth require a separate engineering
measurement plan.

## Separate predictive acceptance

The default scorer refuses a bundle that fails the frozen `cnc-teaching-v1`
acceptance policy or lacks checksummed metrics. `allow_failed_demo=True` is
only for inspecting this failed synthetic model; returned decisions retain
`predictive_acceptance_passed: false`.

After training, inspect `execution_status.json` and `predictive_acceptance.json`:

```bash
python -m sqm_ai.cnc.gate --metrics artifacts/build2/metrics.json \
  --bundle artifacts/build2/bundle.json \
  --report artifacts/build2/predictive_acceptance.json
```

This command exits nonzero for the current synthetic model. Training's zero
exit means software execution completed, not that predictive acceptance passed.
The frozen educational policy checks stage-1 AP >=0.20, recall >=0.90,
flag fraction <=0.05; full-cascade recall >=0.70, precision >=0.20 and
flag fraction <=100/3800. It rejects undefined precision, inconsistent counts,
non-finite metrics and fewer than 32 test failures. These are illustrative
rejection floors, not established business requirements; 32 events is a
demonstration floor, not a statistically justified sample size. A passing
result still grants no production approval. Reports bind metrics and optional
bundle bytes by SHA-256. There is no automatic CNC promotion service.
