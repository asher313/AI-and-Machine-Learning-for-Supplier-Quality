# src/sqm_ai/car/state.py
from typing import Annotated, TypedDict
import operator


class CarState(TypedDict, total=False):
    ncr_id: str
    supplier_id: str
    ncr: dict                    # the NCR row
    evidence: dict               # researcher output
    problem_statement: str
    impact: str
    root_causes: list            # RootCauseSet.hypotheses
    containment: str
    corrective_actions: list
    verification_plan: str
    draft: str                   # assembled template
    failures: list               # this pass only; replaced
    warnings: Annotated[list, operator.add]
    retries: dict                # node name -> attempts
    escalation: str | None       # set when we give up
