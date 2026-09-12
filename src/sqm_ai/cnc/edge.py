# src/sqm_ai/cnc/edge.py
"""Local two-stage suggestions from a trusted, checksummed artifact bundle."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from scipy.special import expit

from sqm_ai.cnc.features import cycle_features, frame
from sqm_ai.cnc.preprocessing import prepare_window, transform


class CycleScorer:
    def __init__(self, artifact_dir):
        folder = Path(artifact_dir)
        self.config = json.loads(
            (folder / "bundle.json").read_text()
        )
        for name, digest in self.config["sha256"].items():
            if (
                Path(name).name != name
                or hashlib.sha256(
                    (folder / name).read_bytes()
                ).hexdigest()
                != digest
            ):
                raise ValueError("artifact checksum mismatch")
        # Only load bundles from a trusted source; hashes do not confer trust.
        self.prep = joblib.load(
            folder / "preprocessing.joblib"
        )
        self.s1 = ort.InferenceSession(
            str(folder / "stage1.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.s2 = ort.InferenceSession(
            str(folder / "stage2.onnx"),
            providers=["CPUExecutionProvider"],
        )

    def score(
        self, window: np.ndarray, context: dict
    ) -> dict:
        cfg = self.config
        prepared = prepare_window(
            window, cfg["channel_mean"], cfg["channel_scale"]
        )
        row = frame([cycle_features(window, context)])
        x = transform(self.prep, row)
        p1 = float(self.s1.run(None, {"input": x})[1][0, 1])
        if not np.isfinite(p1) or not 0 <= p1 <= 1:
            raise ValueError("invalid stage-1 score")
        result = {
            "model_version": cfg["model_version"],
            "stage1_score": p1,
            "stage2_score": None,
            "decision": "no_model_flag",
        }
        if p1 < cfg["stage1_threshold"]:
            return result
        logit = float(
            self.s2.run(None, {"window": prepared[None]})[
                0
            ].reshape(-1)[0]
        )
        if not np.isfinite(logit):
            raise ValueError("invalid stage-2 logit")
        p2 = float(expit(logit))
        result["stage2_score"] = p2
        if p2 >= cfg["stage2_threshold"]:
            result["decision"] = "qa_hold"
        return result
