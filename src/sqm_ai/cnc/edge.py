# sqm_ai/cnc/edge.py
import numpy as np
import onnxruntime as ort

from sqm_ai.cnc.features import cycle_features, frame

STAGE1_THRESHOLD = 0.038
STAGE2_THRESHOLD = 0.31


def fit_window(x: np.ndarray, n: int = 512) -> np.ndarray:
    """Pad with the edge value, or truncate, to n samples."""
    if x.shape[1] >= n:
        return x[:, :n]
    pad = n - x.shape[1]
    return np.pad(x, ((0, 0), (0, pad)), mode="edge")


class CycleScorer:
    """Two-stage scoring for one cell, no network needed."""

    def __init__(
        self,
        stage1_path: str,
        stage2_path: str,
        model_version: str,
    ):
        self.s1 = ort.InferenceSession(stage1_path)
        self.s2 = ort.InferenceSession(stage2_path)
        self.model_version = model_version

    def score(
        self, window: np.ndarray, context: dict
    ) -> dict:
        row = frame([cycle_features(window, context)])
        x = row.select_dtypes("number").to_numpy(
            dtype=np.float32
        )
        p1 = float(
            self.s1.run(None, {"input": x})[1][0][1]
        )
        result = {
            "model_version": self.model_version,
            "stage1_prob": p1,
            "stage2_prob": None,
            "decision": "pass",
        }
        if p1 < STAGE1_THRESHOLD:
            return result
        win = fit_window(window)[None].astype(np.float32)
        logit = float(
            self.s2.run(None, {"window": win})[0][0]
        )
        p2 = 1.0 / (1.0 + np.exp(-logit))
        result["stage2_prob"] = p2
        if p2 >= STAGE2_THRESHOLD:
            result["decision"] = "qa_hold"
        return result
