import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from sqm_ai.build1 import score_suppliers as scoring


class Regressor:
    def predict(self, X):
        return np.array([-2.0, 3.0, 3.0, 9.0])


class Classifier:
    def predict_proba(self, X):
        p = np.array([0.29996, 0.3, 0.99, 0.99])
        return np.column_stack([1 - p, p])


def test_threshold_uses_unrounded_probability_and_support_flags(
    monkeypatch,
):
    features = pd.DataFrame(
        {
            "supplier_id": [
                "S-0001",
                "S-0002",
                "S-0003",
                "S-0004",
            ],
            "ncr_count_90d": [1, 1, 1, 0],
            "units_received_90d": [10, 10, 10, 0],
            "limited_history": [False, False, True, False],
            "snapshot_cutoff": pd.Timestamp("2026-09-01"),
        }
    )
    cols = ["ncr_count_90d", "units_received_90d"]
    metadata = {
        "model_version": "test",
        "training_data_complete_through": "2026-09-01",
    }
    monkeypatch.setattr(
        scoring,
        "load_artifacts",
        lambda _: (
            Regressor(),
            Classifier(),
            cols,
            0.3,
            metadata,
        ),
    )
    monkeypatch.setattr(
        scoring, "build_features", lambda **_: features
    )

    class Explainer:
        def shap_values(self, X):
            return np.zeros((len(X), 2))

    monkeypatch.setattr(
        scoring, "explainer_for", lambda _: Explainer()
    )
    monkeypatch.setattr(
        scoring, "transformed_frame", lambda _, X: X
    )
    monkeypatch.setattr(
        scoring, "top_drivers", lambda *a, **k: []
    )
    result = scoring.score(date(2026, 9, 1))
    assert result.escalate.tolist() == [False, True, False, False]
    assert result.predicted_harm_90d.tolist() == [
        0.0,
        3.0,
        3.0,
        9.0,
    ]
    assert result.harm_was_clipped.tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert result.risk_score.iloc[1] == result.risk_score.iloc[2]
    assert result.risk_score.between(0, 100).all()
    with pytest.raises(ValueError, match="unavailable"):
        scoring.score(date(2026, 8, 1))


def test_corrupt_artifact_fails_before_deserialization(
    tmp_path, monkeypatch
):
    (tmp_path / "risk_regressor.joblib").write_bytes(b"corrupt")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "artifact_sha256": {
                    "risk_regressor.joblib": "bad-hash"
                }
            }
        )
    )
    monkeypatch.setattr(
        scoring.joblib,
        "load",
        lambda *_: pytest.fail("must verify before loading"),
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        scoring.load_artifacts(tmp_path)
