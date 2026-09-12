# Chapter 15 teaching listing. Supply the inputs described in the text.
from pydantic import BaseModel
from sqm_ai.llm import MODELS
from sqm_ai.structured import NCRClassification, classify

class CascadeResult(BaseModel):
    suggestion: NCRClassification
    model: str
    needs_review: bool = True


def classify_cascaded(description):
    """Illustrative routing cutoffs; suggestions always await human disposition."""
    for tier, cutoff in [
        ("fast", 0.90),
        ("standard", 0.85),
        ("frontier", 0.85),
    ]:
        model = MODELS[tier]
        result = classify(description, model=model)
        if result.confidence >= cutoff:
            return CascadeResult(suggestion=result, model=model)
    return CascadeResult(suggestion=result, model=model)
