# Appendix D — D.7 The Structured-Output Call (Chapter 15.3)
from typing import Literal

from pydantic import BaseModel, Field

from sqm_ai.llm import MODELS, client

Category = Literal[
    "cosmetic", "dimensional", "material", "functional"
]
Disposition = Literal[
    "use-as-is", "rework", "scrap", "return-to-supplier"
]


class NCRClassification(BaseModel):
    category: Category
    severity: int = Field(ge=1, le=5)
    suggested_disposition: Disposition
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=20)


SYSTEM = (
    "You are an aerospace supplier-quality engineer. "
    "Classify the nonconformance report the user gives you. "
    "Severity 5 is "
    "safety-critical; 1 is trivial. Confidence is your honest "
    "probability that category and severity are both right."
)


def classify(description: str) -> NCRClassification:
    response = client.messages.parse(
        model=MODELS["standard"],
        max_tokens=512,
        system=SYSTEM,
        messages=[{"role": "user", "content": description}],
        output_format=NCRClassification,
        output_config={"effort": "low"},
    )
    return response.parsed_output
