import math

import pytest

from sqm_ai.assistant.metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_metrics_match_hand_calculations_and_candidate_vs_final_recall():
    ranking = [3, 1, 2]
    assert recall_at_k(ranking, {1, 2}, 3) == 1
    assert recall_at_k(ranking, {1, 2}, 1) == 0
    assert reciprocal_rank(ranking, {1, 2}) == 0.5
    assert ndcg_at_k(ranking, {1: 2, 2: 1}, 3) == pytest.approx(
        (2 / math.log2(3) + 1 / math.log2(4))
        / (2 + 1 / math.log2(3))
    )
    assert ndcg_at_k([1, 2], {1: 2, 2: 1}, 2) == 1


def test_invalid_and_unanswerable_inputs_are_not_silent_zeroes():
    with pytest.raises(ValueError):
        recall_at_k([1], set(), 1)
    with pytest.raises(ValueError):
        reciprocal_rank([1], set())
    with pytest.raises(ValueError):
        ndcg_at_k([1], {1: 0}, 1)
    with pytest.raises(ValueError):
        ndcg_at_k([1, 1], {1: 2}, 2)
    with pytest.raises(ValueError):
        ndcg_at_k([1], {1: -1}, 1)
    with pytest.raises(ValueError):
        recall_at_k([1], {1}, 0)
