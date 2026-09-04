# Chapter 15 — 15.9 Model Cascades
from sqm_ai.llm import MODELS, client

# SYSTEM and NCRClassification: imports as in §15.3.


def classify_with(
    model: str, description: str,
) -> NCRClassification:
    response = client.messages.parse(
        model=model, max_tokens=512, system=SYSTEM,
        messages=[{"role": "user", "content": description}],
        output_format=NCRClassification,
        output_config={"effort": "low"},
    )
    return response.parsed_output


def classify_cascaded(description: str) -> NCRClassification:
    """Cheap first; escalate only when the model is unsure."""
    result = classify_with(MODELS["fast"], description)
    if result.confidence >= 0.90:
        return result
    result = classify_with(MODELS["standard"], description)
    if result.confidence >= 0.85:
        return result
    return classify_with(MODELS["frontier"], description)
