"""Contracts for constructed examples, not model quality gates."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from sqm_ai.cnc.features import cycle_features
from sqm_ai.cnc.synthetic import batch_features, generate
from sqm_ai.ncr import NCRInput
from sqm_ai.stats import bootstrap_ci, bootstrap_diff_ci
from sqm_ai.synthetic import cobalt_lots, suppliers


def test_onboarding_explains_canonical_labeled_count():
    s = suppliers()
    months = pd.date_range("2023-09-01", periods=33, freq="MS")
    assert len(s) == s.supplier_id.nunique() == 1800
    assert (
        sum(int((months >= d).sum()) for d in s.onboarded_at)
        == 57118
    )


def test_constructed_lots_match_every_printed_statistic():
    lots = cobalt_lots()
    a = lots.loc[lots.period.eq("before"), "fpy"].to_numpy()
    b = lots.loc[lots.period.eq("after"), "fpy"].to_numpy()
    assert (len(a), len(b)) == (41, 38)
    assert lots.fpy.between(0, 1).all()
    assert [
        round(v, 4)
        for v in [
            a.mean(),
            a.std(ddof=1),
            b.mean(),
            b.std(ddof=1),
        ]
    ] == [0.9502, 0.0271, 0.9202, 0.0378]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    assert (round(t, 2), round(p, 5)) == (4.03, 0.00015)
    assert round(stats.mannwhitneyu(a, b).pvalue, 5) == 0.00047
    assert tuple(np.round(bootstrap_ci(b, np.mean), 4)) == (
        0.9083,
        0.9320,
    )
    assert tuple(
        np.round(bootstrap_diff_ci(a, b, np.mean), 4)
    ) == (-0.0446, -0.0161)


def test_ncr_ids_support_the_books_annual_volume():
    data = {
        "ncr_id": "NCR-2026-29412",
        "supplier_id": "S-0417",
        "defect_description": "Synthetic dimensional error",
        "quantity": 1,
        "discovered_at": "2026-08-01T12:00:00",
    }
    assert NCRInput(**data).ncr_id == data["ncr_id"]
    with pytest.raises(ValueError):
        NCRInput(**{**data, "ncr_id": "NCR-2026-29412\n"})


def test_streaming_cnc_features_equal_reference_extractor():
    rng = np.random.default_rng(88)
    windows = rng.normal(size=(4, 5, 480)).astype(np.float32)
    windows[:, 2] += 2
    windows[:, 3] = 180 + windows[:, 3]
    windows[1, 0] = 0  # constant sensor must remain finite
    context = pd.DataFrame(
        [
            {
                "nominal_seconds": 48,
                "tool_life_pct": 60,
                "prior5_fails": 1,
                "part_family": "fitting",
                "machine_id": "TUL-CNC-07",
                "shift": "1",
            }
        ]
        * 4
    )
    batch = batch_features(windows, context)
    for i, window in enumerate(windows):
        reference = cycle_features(
            window.astype(float), context.iloc[i].to_dict()
        )
        assert len(reference) == len(batch.columns) == 51
        for col, value in reference.items():
            if isinstance(value, str):
                assert batch.loc[i, col] == value
            else:
                assert batch.loc[i, col] == pytest.approx(
                    value, abs=1e-9
                )


def test_cnc_generation_is_independent_of_batch_size(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate(a, cycles=1000, raw_cycles=100, batch_size=128)
    generate(b, cycles=1000, raw_cycles=100, batch_size=333)
    pd.testing.assert_frame_equal(
        pd.read_parquet(a / "cycle_features.parquet"),
        pd.read_parquet(b / "cycle_features.parquet"),
    )
    np.testing.assert_array_equal(
        np.load(a / "windows.npy"), np.load(b / "windows.npy")
    )
    with pytest.raises(FileExistsError):
        generate(a, cycles=1000)
    df = pd.read_parquet(a / "cycle_features.parquet")
    meta = pd.read_parquet(a / "cycles.parquet")
    assert df.failed.sum() == 6
    for i in [0, 451, 800, 999]:
        available = meta[
            meta.machine_id.eq(df.loc[i, "machine_id"])
            & meta.inspection_completed_at.lt(
                df.loc[i, "cycle_start"]
            )
        ]
        assert (
            df.loc[i, "prior5_fail_count"]
            == available.tail(5).failed.sum()
        )
