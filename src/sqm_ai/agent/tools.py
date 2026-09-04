# src/sqm_ai/agent/tools.py
"""Tools for the NCR agent: one reused, two new."""
from sqlalchemy import create_engine, text

from sqm_ai.settings import get_settings
from sqm_ai.triage.classify import TRIAGE_TOOL   # Chapter 16
from sqm_ai.triage.schema import TriageResult

_engine = create_engine(get_settings().database_url)

HISTORY_SQL = """
SELECT category, severity, disposition, discovered_at
  FROM sqm.ncrs
 WHERE supplier_id = :sid
   AND discovered_at > current_date - INTERVAL '90 days'
 ORDER BY discovered_at DESC
 LIMIT 10
"""

HISTORY_TOOL = {
    "name": "get_supplier_history",
    "description": (
        "Return this supplier's nonconformances over the "
        "last 90 days, most recent first."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "supplier_id": {
                "type": "string",
                "pattern": r"^S-\d{4}$",
            },
        },
        "required": ["supplier_id"],
        "additionalProperties": False,
    },
}

CAR_TOOL = {
    "name": "draft_car",
    "description": (
        "Open a corrective action request draft. Use only "
        "when severity is 3 or higher."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "ncr_id": {"type": "string"},
            "supplier_id": {"type": "string"},
            "root_cause_hypothesis": {"type": "string"},
            "proposed_containment": {"type": "string"},
        },
        "required": [
            "ncr_id", "supplier_id",
            "root_cause_hypothesis", "proposed_containment",
        ],
        "additionalProperties": False,
    },
}


def record_triage(**kwargs) -> str:
    """Chapter 16's tool, called as a step of the agent."""
    result = TriageResult.model_validate(kwargs)
    return (
        f"Recorded {result.category}, severity "
        f"{result.severity}, disposition "
        f"{result.suggested_disposition}."
    )


def get_supplier_history(supplier_id: str) -> str:
    with _engine.connect() as conn:
        rows = conn.execute(
            text(HISTORY_SQL), {"sid": supplier_id}
        ).fetchall()
    if not rows:
        return f"{supplier_id}: no NCRs in 90 days."
    lines = [
        f"{d:%Y-%m-%d} {c} sev {s} -> {disp}"
        for c, s, disp, d in rows
    ]
    return (
        f"{supplier_id}: {len(rows)} NCRs in 90 days\n"
        + "\n".join(lines)
    )


def draft_car(
    ncr_id: str, supplier_id: str,
    root_cause_hypothesis: str, proposed_containment: str,
) -> str:
    """Write a CAR stub row; the human owns the document."""
    # Chapter 20 replaces this stub with the drafting crew.
    return f"CAR draft opened for {ncr_id} ({supplier_id})."


TOOLS = [TRIAGE_TOOL, HISTORY_TOOL, CAR_TOOL]

IMPLS = {
    "record_triage": record_triage,
    "get_supplier_history": get_supplier_history,
    "draft_car": draft_car,
}
