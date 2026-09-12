"""Validated input, model suggestion, and routing outcome."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal[
    "cosmetic", "dimensional", "material", "functional"
]
Disposition = Literal[
    "use-as-is", "rework", "scrap", "return-to-supplier"
]


class TriageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Category
    severity: int = Field(ge=1, le=5)
    supplier_id: str = Field(pattern=r"^S-\d{4}$")
    suggested_disposition: Disposition
    confidence: float = Field(ge=0, le=1)
    reasoning: str = Field(min_length=20, max_length=600)


class NCRInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ncr_id: str = Field(min_length=1)
    part_number: str = Field(min_length=1)
    quantity: int = Field(ge=1)
    defect_description: str = Field(min_length=1)
    supplier_shortlist: list[str] = Field(min_length=1)
    data_classification: Literal[
        "synthetic",
        "approved_uncontrolled",
        "restricted",
        "unknown",
    ] = "unknown"


class TriageDecision(BaseModel):
    decision_id: str
    ncr_id: str
    result: TriageResult | None = None
    model: str = "none"
    stage: str = "review"
    routed: Literal["auto", "review"] = "review"
    review_notes: str | None = None
    rules_fired: list[str] = Field(default_factory=list)
    trace_ids: list[str] = Field(default_factory=list)
    car_owner_email: str | None = None
