# sqm_ai/cnc/export.py
import numpy as np
import onnxruntime as ort
import torch
from onnxmltools.convert import convert_xgboost
from onnxconverter_common.data_types import FloatTensorType

from sqm_ai.dl.cnc_cnn import CNCWindowNet


def export_stage1(model, n_features: int, path: str):
    initial = [("input", FloatTensorType([None,
                                          n_features]))]
    onx = convert_xgboost(
        model,
        initial_types=initial,
        # plain arrays out, not a class->prob dict
        options={id(model): {"zipmap": False}},
    )
    with open(path, "wb") as fh:
        fh.write(onx.SerializeToString())


def export_stage2(model: CNCWindowNet, path: str):
    model.eval()
    dummy = torch.randn(1, 5, 512)
    torch.onnx.export(
        model, (dummy,), path,
        input_names=["window"],
        output_names=["logit"],
        dynamic_axes={
            "window": {0: "batch"},
            "logit": {0: "batch"},
        },
    )


def assert_parity(model, path: str, sample: np.ndarray):
    """The exported model must match the trained one."""
    model.eval()
    with torch.no_grad():
        want = model(
            torch.tensor(sample, dtype=torch.float32)
        ).numpy()
    sess = ort.InferenceSession(path)
    got = sess.run(
        None, {sess.get_inputs()[0].name:
               sample.astype(np.float32)}
    )[0].squeeze(-1)
    np.testing.assert_allclose(
        want, got, rtol=1e-5, atol=1e-5
    )
