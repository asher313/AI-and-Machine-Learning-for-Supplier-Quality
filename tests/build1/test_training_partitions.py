import numpy as np
import pandas as pd
import pytest

from sqm_ai.build1.features import load_modelling_frame
from sqm_ai.build1.train import split_train_tail
from sqm_ai.features import NUMERIC, month_folds


def panel():
    months = pd.date_range("2023-09-01", periods=33, freq="MS")
    df = pd.DataFrame({"month": np.repeat(months, 3)})
    df["label_end"] = (
        df.month
        + pd.offsets.MonthBegin(1)
        + pd.Timedelta(days=90)
    )
    df["data_complete_through"] = pd.Timestamp("2026-09-01")
    df["sev3_next_90d"] = np.arange(len(df)) % 2
    df["harm_next_90d"] = df.sev3_next_90d * 4
    for col in NUMERIC:
        df[col] = 1.0
    df["tier"] = "B"
    df.index = (
        1000 + np.arange(len(df)) * 7
    )  # labels are not positions
    return df


def test_calibration_is_disjoint_with_mature_inner_labels():
    df = panel()
    for train, test in month_folds(df):
        fit, cal = split_train_tail(df, train)
        assert not set(fit) & set(cal)
        assert not set(train) & set(test)
        assert len(set(df.iloc[cal].month)) == 3
        cutoff = df.iloc[cal].month.min() + pd.offsets.MonthBegin(
            1
        )
        assert df.iloc[fit].label_end.le(cutoff).all()
        assert df.iloc[fit].month.max() < df.iloc[cal].month.min()


def test_training_loader_excludes_targets_and_checks_maturity():
    df = panel()
    X, yc, yr = load_modelling_frame(df)
    assert yc.equals(df.sev3_next_90d.astype(int))
    assert yr.equals(df.harm_next_90d.astype(float))
    assert list(X.columns) == NUMERIC + ["tier"]
    assert X.index.equals(df.index)
    invalid = df.copy()
    invalid.loc[invalid.index[-1], "data_complete_through"] = (
        pd.Timestamp("2025-01-01")
    )
    with pytest.raises(ValueError, match="complete"):
        load_modelling_frame(invalid)
    invalid = df.copy()
    invalid.loc[invalid.index[-1], "sev3_next_90d"] = np.nan
    with pytest.raises(ValueError, match="censored"):
        load_modelling_frame(invalid)
