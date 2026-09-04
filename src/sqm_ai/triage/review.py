# src/sqm_ai/triage/review.py
import random

KNOWN_ANSWER_RATE = 0.05


def enqueue(row: dict, golden: list[dict]) -> list[dict]:
    """The real row, sometimes followed by a probe drawn
    from the golden set. Reviewers see no difference."""
    rows = [row]
    if golden and random.random() < KNOWN_ANSWER_RATE:
        probe = dict(random.choice(golden))
        probe["is_known_answer"] = True
        probe["routed"] = "review"
        rows.append(probe)
    return rows
