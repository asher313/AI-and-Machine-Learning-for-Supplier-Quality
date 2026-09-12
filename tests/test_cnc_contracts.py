"""Guard against skew, gate tie assumptions, and leakage in Build 2."""
import numpy as np
import pandas as pd
import pytest

from sqm_ai.cnc.preprocessing import (
    channel_statistics,
    prepare_window,
)
from sqm_ai.cnc.train_stage1 import (
    fit_stage1,
    threshold_for_flag_rate,
)
from sqm_ai.cnc.train_stage2 import fit_stage2


def test_normalization_ignores_padding_and_rejects_truncation():
    windows = np.full((2, 5, 512), 9999., dtype=np.float32)
    windows[0, :, :4] = 1
    windows[1, :, :4] = 3
    mean, scale = channel_statistics(windows, [4, 4])
    np.testing.assert_allclose(mean, 2)
    np.testing.assert_allclose(scale, 1)
    ready = prepare_window(windows[0, :, :4], mean, scale)
    assert ready.shape == (5, 512)
    np.testing.assert_allclose(ready[:, :4], -1)
    assert not ready[:, 4:].any()
    with pytest.raises(ValueError, match="longer cycles"):
        prepare_window(np.ones((5, 600)), mean, scale)


def test_gate_quantile_does_not_promise_exact_rate():
    scores = np.ones(20)*0.4
    threshold = threshold_for_flag_rate(scores, 0.05)
    assert (scores >= threshold).mean() == 1
    with pytest.raises(ValueError):
        threshold_for_flag_rate([np.nan], .05)


def test_training_rejects_future_inspection_and_one_class(tmp_path):
    fit = pd.DataFrame({"cycle_start": pd.to_datetime(["2026-01-01"]), "inspection_completed_at": pd.to_datetime(["2026-02-01"])})
    cal = pd.DataFrame({"cycle_start": pd.to_datetime(["2026-01-02"])})
    with pytest.raises(ValueError, match="available"):
        fit_stage1(fit, cal)
    with pytest.raises(ValueError, match="both training classes"):
        fit_stage2(np.ones((4,5,512)), np.zeros(4), 2, np.full(4,480), ckpt_path=str(tmp_path / "bad.pt"))
