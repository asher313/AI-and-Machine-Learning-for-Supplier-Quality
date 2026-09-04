# src/sqm_ai/triage/schema.py
from typing import Literal
from pydantic import BaseModel, Field

Category = Literal[
    "cosmetic", "dimensional", "material", "functional"]
Disposition = Literal[
    "use-as-is", "rework", "scrap", "return-to-supplier"]


class TriageResult(BaseModel):
    """What the classifier is allowed to say."""
    category: Category
    severity: int = Field(ge=1, le=5)
    supplier_id: str = Field(pattern=r"^S-\d{4}$")
    suggested_disposition: Disposition
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=20, max_length=600)
