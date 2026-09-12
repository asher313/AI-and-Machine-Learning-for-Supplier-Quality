import hashlib
import json
from pathlib import Path
import pytest
from sqm_ai.ci.gate import assess
from sqm_ai.ci.pipeline import register, rehearse


def metrics():
    return [
        {
            "fold": 1,
            "n_test": 100,
            "positive_test": 10,
            "screen_threshold": 0.04,
            "roc_auc": 0.9,
            "recall_sev3": 0.95,
        }
    ]


@pytest.mark.parametrize(
    "value",
    [None, float("nan"), float("inf"), True, 1.1, -0.1, "0.9"],
)
def test_gate_refuses_invalid_metric(value):
    data = metrics()
    data[0]["roc_auc"] = value
    with pytest.raises(ValueError):
        assess(data, 0.85, 0.9)


def test_gate_uses_latest_fold_and_fixed_screen():
    data = metrics()
    data.append(dict(data[0], fold=2, recall_sev3=0.84))
    assert not assess(data, 0.85, 0.9)["passed"]
    data[-1]["screen_threshold"] = 0.3
    with pytest.raises(ValueError):
        assess(data, 0.85, 0.9)
    with pytest.raises(ValueError):
        assess(metrics()[::-1] * 2, 0.85, 0.9)


def test_registration_binds_gate_data_and_artifacts(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    for name in [
        "risk_regressor.joblib",
        "risk_classifier_calibrated.joblib",
        "feature_list.json",
        "threshold.json",
    ]:
        (model / name).write_text(
            "synthetic fixture, never load as pickle"
        )
    raw = json.dumps(metrics())
    (model / "metrics.json").write_text(raw)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in model.iterdir()
    }
    (model / "metadata.json").write_text(
        json.dumps(
            {
                "artifact_sha256": hashes,
                "model_version": "fixture-v1",
                "training_data_sha256": "fixture-data",
            }
        )
    )
    gate = tmp_path / "gate.json"
    report = assess(metrics(), 0.85, 0.9)
    report["metrics_sha256"] = hashes["metrics.json"]
    gate.write_text(json.dumps(report))
    release = register(model, gate, "code-commit", "unique-run")
    assert (
        rehearse(release, model, "staging")["service_deployed"]
        is False
    )
    report["passed"] = False
    gate.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        register(model, gate, "code-commit", "unique-run")
    (model / "risk_regressor.joblib").write_text("changed")
    with pytest.raises(ValueError):
        rehearse(release, model, "production")
