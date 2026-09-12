# src/sqm_ai/cnc/export.py
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import (
    FloatTensorType,
)


def export_stage1(model, n_features: int, path: str):
    """Export the numeric XGBClassifier after persisted preprocessing."""
    onx = convert_xgboost(
        model,
        initial_types=[
            ("input", FloatTensorType([None, n_features]))
        ],
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(onx.SerializeToString())


def export_stage2(model, path: str):
    model = model.cpu().eval()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(2, 5, 512)
    torch.onnx.export(
        model,
        (dummy,),
        path,
        dynamo=True,
        input_names=["window"],
        output_names=["logit"],
        dynamic_shapes=(
            {0: torch.export.Dim("batch", min=1)},
        ),
    )


def assert_stage1_parity(model, path, sample):
    sample = np.asarray(sample, dtype=np.float32)
    sess = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    got = sess.run(None, {"input": sample})[1][:, 1]
    np.testing.assert_allclose(
        model.predict_proba(sample)[:, 1],
        got,
        rtol=1e-5,
        atol=1e-5,
    )


def assert_parity(model, path, sample):
    model = model.cpu().eval()
    sample = np.asarray(sample, dtype=np.float32)
    with torch.no_grad():
        want = (
            model(torch.from_numpy(sample))
            .numpy()
            .reshape(-1)
        )
    sess = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    got = sess.run(None, {"window": sample})[0].reshape(-1)
    np.testing.assert_allclose(
        want, got, rtol=1e-5, atol=1e-5
    )
