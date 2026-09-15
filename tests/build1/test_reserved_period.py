"""Time-boundary and target-maturity checks for the reserved final protocol."""

import numpy as np
import pandas as pd
import pytest

from sqm_ai.build1.evaluation import reserve_period
from sqm_ai.features import NUMERIC


def panel(start="2023-09-01", periods=33):
    months = pd.date_range(start, periods=periods, freq="MS")
    df = pd.DataFrame({"month": np.repeat(months, 2)})
    df["supplier_id"] = ["S-0001", "S-0002"] * len(months)
    df["label_end"] = (
        df.month
        + pd.offsets.MonthBegin(1)
        + pd.Timedelta(days=90)
    )
    df["data_complete_through"] = df.label_end.max()
    df["sev3_next_90d"] = np.arange(len(df)) % 2
    df["harm_next_90d"] = df.sev3_next_90d * 4.0
    for column in NUMERIC:
        df[column] = 1.0
    df["tier"] = "B"
    df.index = 1000 + np.arange(len(df)) * 7
    return df


def test_reservation_uses_prediction_cutoff_and_exact_maturity_boundary():
    df = panel().sample(frac=1, random_state=42)
    original = df.copy(deep=True)
    dev, final, info = reserve_period(df, "2026-03-01")
    # December's forecast ends exactly on April 1; it is available. January
    # and February forecasts have not yet matured and must be purged.
    assert dev.month.max() == pd.Timestamp("2025-12-01")
    assert dev.label_end.max() == pd.Timestamp("2026-04-01")
    assert final.month.min() == pd.Timestamp("2026-03-01")
    assert info["first_prediction_cutoff"] == "2026-04-01"
    assert info["purged_rows"] == 4
    assert len(dev) + len(final) + info["purged_rows"] == len(df)
    assert set(dev.supplier_id) == set(final.supplier_id)
    assert set(dev.month).isdisjoint(set(final.month))
    assert set(dev.index).isdisjoint(set(final.index))
    pd.testing.assert_frame_equal(df, original)


def test_ninety_day_maturity_across_leap_year_is_not_three_calendar_months():
    dev, final, info = reserve_period(
        panel("2023-09-01", 10), "2024-02-01"
    )
    # Dec 1 + 90 days = Feb 29 in a leap year, before the March 1 cutoff.
    assert dev.month.max() == pd.Timestamp("2023-11-01")
    assert dev.label_end.max() == pd.Timestamp("2024-02-29")
    assert final.month.min() == pd.Timestamp("2024-02-01")
    assert info["purged_rows"] == 4


@pytest.mark.parametrize(
    "start",
    [
        "2026-03-02",
        "2026-03-01T01:00:00",
        "2026-03-01T00:00:00Z",
        "NaT",
    ],
)
def test_invalid_final_boundary_rejected(start):
    with pytest.raises(ValueError, match="month start"):
        reserve_period(panel(), start)


@pytest.mark.parametrize(
    "column,value",
    [
        ("sev3_next_90d", np.nan),
        ("harm_next_90d", np.nan),
        ("data_complete_through", pd.Timestamp("2026-04-01")),
    ],
)
def test_final_labels_must_already_be_complete(column, value):
    df = panel()
    df.loc[df.index[-1], column] = value
    with pytest.raises(ValueError, match="censored|complete"):
        reserve_period(df, "2026-03-01")


def test_shortened_target_horizon_cannot_evade_maturity_purge():
    df = panel()
    january = df.month.eq(pd.Timestamp("2026-01-01"))
    df.loc[january, "label_end"] = pd.Timestamp("2026-04-01")
    with pytest.raises(ValueError, match="90 days"):
        reserve_period(df, "2026-03-01")


def test_month_identity_cannot_split_one_month_into_distinct_cohorts():
    df = panel()
    df.loc[df.index[-1], "month"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="month-start identity"):
        reserve_period(df, "2026-03-01")


def test_duplicate_identity_rejected_even_with_unique_index():
    df = panel()
    df = pd.concat([df, df.iloc[[-1]]], ignore_index=True)
    with pytest.raises(
        ValueError, match="duplicate supplier-month"
    ):
        reserve_period(df, "2026-03-01")


@pytest.mark.parametrize("start", ["2023-09-01", "2026-06-01"])
def test_both_partitions_must_be_nonempty(start):
    with pytest.raises(ValueError, match="nonempty"):
        reserve_period(panel(), start)


def make_partition(tmp_path, monkeypatch):
    import sys

    from sqm_ai.build1.evaluation import main

    source = tmp_path / "source.parquet"
    panel().to_parquet(source, index=False)
    partition = tmp_path / "reserved"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluation",
            "reserve",
            "--input",
            str(source),
            "--final-start",
            "2026-03-01",
            "--output",
            str(partition),
        ],
    )
    main()
    return partition


def test_evaluator_rejects_changed_partition_before_loading_models(
    tmp_path, monkeypatch
):
    import sys

    from sqm_ai.build1.evaluation import main

    partition = make_partition(tmp_path, monkeypatch)
    path = partition / "final.parquet"
    changed = pd.read_parquet(path)
    changed.loc[0, "harm_next_90d"] += 100
    changed.to_parquet(path, index=False)
    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluation",
            "evaluate-final",
            "--partition",
            str(partition),
            "--models",
            str(tmp_path / "models"),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(
        ValueError, match="partition checksum mismatch"
    ):
        main()
    assert not output.exists()


def test_evaluator_rejects_model_fitted_on_other_dataset(
    tmp_path, monkeypatch
):
    import sys

    from sqm_ai.build1 import score_suppliers
    from sqm_ai.build1.evaluation import main

    partition = make_partition(tmp_path, monkeypatch)
    monkeypatch.setattr(
        score_suppliers,
        "load_artifacts",
        lambda path: (
            None,
            None,
            [],
            0.3,
            {"training_data_sha256": "different dataset"},
        ),
    )
    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluation",
            "evaluate-final",
            "--partition",
            str(partition),
            "--models",
            str(tmp_path / "models"),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(
        ValueError, match="not fitted to this reserved"
    ):
        main()
    assert not output.exists()


def test_equal_size_swapped_partitions_do_not_evade_separation(
    tmp_path, monkeypatch
):
    import json
    import sys

    from sqm_ai.build1.evaluation import digest, main

    partition = make_partition(tmp_path, monkeypatch)
    dev = pd.read_parquet(partition / "development.parquet")
    final = pd.read_parquet(partition / "final.parquet")
    original_dev_row = dev.iloc[[-1]].copy()
    original_final_row = final.iloc[[0]].copy()
    dev = pd.concat(
        [dev.iloc[:-1], original_final_row], ignore_index=True
    )
    final = pd.concat(
        [original_dev_row, final.iloc[1:]], ignore_index=True
    )
    manifest = json.loads((partition / "split.json").read_text())
    for name, df in (("development", dev), ("final", final)):
        df.to_parquet(partition / f"{name}.parquet", index=False)
        manifest[f"{name}_sha256"] = digest(
            partition / f"{name}.parquet"
        )
    (partition / "split.json").write_text(json.dumps(manifest))
    output = tmp_path / "evaluation"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluation",
            "evaluate-final",
            "--partition",
            str(partition),
            "--models",
            str(tmp_path / "models"),
            "--output",
            str(output),
        ],
    )
    # Must fail before model loading, although all counts and checksums agree.
    with pytest.raises(
        ValueError, match="time/label-maturity separation"
    ):
        main()
    assert not output.exists()


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_nonfinite_forward_burden_rejected_before_training(value):
    df = panel()
    df.loc[df.index[-1], "harm_next_90d"] = value
    with pytest.raises(ValueError, match="finite"):
        reserve_period(df, "2026-03-01")
