import numpy as np
import pandas as pd
import pytest

from sqm_ai.build1.evaluation import monthly_policy_report


def frame(n=100):
    return pd.DataFrame(
        {
            "month": pd.Timestamp("2026-01-01"),
            "supplier_id": [f"S-{i:04}" for i in range(n)],
            "predicted_harm_90d": np.arange(n, dtype=float),
            "recent_harm_90d": np.arange(n, dtype=float),
            "sev3_next_90d": (np.arange(n) >= n - 50).astype(int),
            "harm_next_90d": np.arange(n, dtype=float),
        }
    )


def test_global_rank_does_not_replace_actual_queue_metrics():
    df = frame(1800)
    # Exchange ranks of the 50 serious-event suppliers with the next 50.
    df.loc[1700:1749, "predicted_harm_90d"] = np.arange(
        1750, 1800
    )
    df.loc[1750:1799, "predicted_harm_90d"] = np.arange(
        1700, 1750
    )
    result = monthly_policy_report(df)["rows"]
    assert result[0]["precision_at_k"] == 0
    assert result[0]["recall_at_k"] == 0
    assert result[1]["precision_at_k"] == 1
    assert len(result[0]["selected_ids"]) == 50


def test_ties_are_deterministic_and_denominators_are_monthly():
    df = frame(4)
    df["predicted_harm_90d"] = 1.0
    other = df.copy()
    other["month"] = pd.Timestamp("2026-02-01")
    other["sev3_next_90d"] = 0
    other["harm_next_90d"] = 0
    report = monthly_policy_report(
        pd.concat([other, df]).sample(frac=1), k=2
    )
    jan, _, feb, _ = report["rows"]
    assert jan["selected_ids"] == ["S-0000", "S-0001"]
    assert jan["selected_suppliers"] == 2
    assert feb["precision_at_k"] == 0
    assert feb["recall_at_k"] is None
    assert feb["captured_harm_fraction"] is None


@pytest.mark.parametrize(
    "column,value",
    [
        ("predicted_harm_90d", np.inf),
        ("harm_next_90d", -1),
        ("sev3_next_90d", 0.5),
        ("recent_harm_90d", np.nan),
    ],
)
def test_invalid_policy_evidence_rejected(column, value):
    df = frame().astype({"sev3_next_90d": float})
    df.loc[0, column] = value
    with pytest.raises(ValueError):
        monthly_policy_report(df)


def test_duplicate_candidate_rejected():
    df = frame()
    with pytest.raises(ValueError, match="duplicate"):
        monthly_policy_report(pd.concat([df, df.iloc[:1]]))


def test_small_cohort_uses_actual_selected_count_not_requested_capacity():
    df = frame(3)
    df["sev3_next_90d"] = [1, 0, 1]
    df["harm_next_90d"] = [3.0, 0.0, 8.0]
    for row in monthly_policy_report(df, k=50)["rows"]:
        assert row["eligible_suppliers"] == 3
        assert row["selected_suppliers"] == 3
        assert row["precision_at_k"] == pytest.approx(2 / 3)
        assert row["recall_at_k"] == 1
        assert row["captured_harm_fraction"] == 1


def test_policy_metrics_use_same_monthly_cohort_but_distinct_rankings():
    df = frame(4)
    df["predicted_harm_90d"] = [4, 3, 2, 1]
    df["recent_harm_90d"] = [1, 2, 3, 4]
    df["sev3_next_90d"] = [1, 0, 1, 1]
    df["harm_next_90d"] = [4.0, 1.0, 6.0, 9.0]
    model, baseline = monthly_policy_report(df, k=2)["rows"]
    assert model["precision_at_k"] == 0.5
    assert model["recall_at_k"] == pytest.approx(1 / 3)
    assert model["captured_harm_fraction"] == 0.25
    assert baseline["precision_at_k"] == 1
    assert baseline["recall_at_k"] == pytest.approx(2 / 3)
    assert baseline["captured_harm_fraction"] == 0.75
    assert model["harm_total"] == baseline["harm_total"] == 20
    assert (
        model["serious_events_total"]
        == baseline["serious_events_total"]
        == 3
    )
