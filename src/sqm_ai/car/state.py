"""JSON-serializable workflow state plus validated role contracts."""

import operator
import hashlib
import json
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ncr_id: str
    supplier_id: str
    sources: list[EvidenceItem] = Field(min_length=1)
    investigation_gaps: list[str]

    @model_validator(mode="after")
    def unique_ids(self):
        if len({s.id for s in self.sources}) != len(self.sources):
            raise ValueError("unique evidence IDs required")
        return self


class ProblemSections(BaseModel):
    model_config = ConfigDict(extra="forbid")
    problem_statement: str = Field(min_length=10)
    impact: str = Field(min_length=10)
    evidence_refs: list[str] = Field(min_length=1)


class ActionSections(BaseModel):
    model_config = ConfigDict(extra="forbid")
    containment: str = Field(min_length=10)
    corrective_actions: list[str] = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    investigation_gaps: list[str]


class VerificationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_size: int = Field(ge=1)
    duration_days: int = Field(ge=1)
    metric: str = Field(min_length=3)
    acceptance_criterion: str = Field(min_length=5)
    design_basis: str = Field(min_length=10)
    evidence_refs: list[str] = Field(min_length=1)
    investigation_gaps: list[str]


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actor_id: str
    draft_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    action: Literal["accept_draft", "reject", "provide_evidence"]
    note: str = Field(min_length=1)
    evidence: EvidenceBundle | None = None

    @model_validator(mode="after")
    def evidence_action(self):
        if (self.action == "provide_evidence") != (
            self.evidence is not None
        ):
            raise ValueError(
                "provide_evidence requires evidence; other actions do not"
            )
        return self


class CarState(TypedDict, total=False):
    ncr_id: str
    supplier_id: str
    evidence: dict
    run_scope: dict
    problem: dict
    root_cause: dict
    actions: dict
    verification_plan: dict
    draft: str
    failures: list[dict]
    warnings: Annotated[list[str], operator.add]
    retries: dict[str, int]
    retry_target: str | None
    gaps: list[str]
    role_errors: dict[str, str]
    status: str
    review_evidence: dict | None
    reviews: Annotated[list[dict], operator.add]


def draft_revision(state):
    """Bind a decision to the displayed draft, evidence and immutable run scope."""
    payload = {
        key: state.get(key)
        for key in (
            "draft",
            "evidence",
            "run_scope",
            "failures",
            "gaps",
        )
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
