import numpy as np
import torch
import xgboost as xgb

from sqm_ai.cnc.export import (
    assert_parity,
    assert_stage1_parity,
    export_stage1,
    export_stage2,
)
from sqm_ai.dl.cnc_cnn import CNCWindowNet


def test_xgboost_onnx_parity(tmp_path):
    x = np.random.default_rng(42).normal(size=(100,4)).astype(np.float32)
    y = (x[:,0] > 0).astype(int)
    model = xgb.XGBClassifier(n_estimators=5, n_jobs=1).fit(x,y)
    path = tmp_path / "xgb.onnx"
    export_stage1(model, 4, str(path))
    assert_stage1_parity(model, path, x)


def test_cnn_dynamic_batch_parity(tmp_path):
    torch.manual_seed(42)
    model = CNCWindowNet().eval()
    path = tmp_path / "cnn.onnx"
    export_stage2(model, str(path))
    for count in [1,3]:
        sample = np.random.default_rng(count).normal(size=(count,5,512)).astype(np.float32)
        assert_parity(model, path, sample)
