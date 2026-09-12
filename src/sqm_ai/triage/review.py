"""Separate evaluation tasks; probes never share a live NCR primary key."""

import random
import uuid
from copy import deepcopy

KNOWN_ANSWER_RATE = 0.05


def enqueue(row, probes, *, rng=None):
    """Return a live review task and occasionally an isolated probe task.

    A separate store owns probe truth; do not send it to the reviewer UI.
    At p=q/(1-q), expected probes/(real+probes) approaches q.
    """
    rng = rng or random
    if row["routed"] != "review":
        return []
    tasks = [
        {
            "task_id": str(uuid.uuid4()),
            "kind": "live",
            "decision_id": row["decision_id"],
        }
    ]
    if probes and rng.random() < KNOWN_ANSWER_RATE / (
        1 - KNOWN_ANSWER_RATE
    ):
        probe = rng.choice(probes)
        tasks.append(
            {
                "task_id": str(uuid.uuid4()),
                "kind": "probe",
                "probe_id": probe["probe_id"],
                "candidate": deepcopy(probe["candidate"]),
            }
        )
    return tasks
