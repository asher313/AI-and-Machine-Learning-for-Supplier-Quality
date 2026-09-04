# sqm_ai/dl/ncr_zero_shot.py
from sqm_ai.llm import MODELS, client

SYSTEM = (
    "You classify aerospace nonconformance reports.\n"
    "Category is exactly one of: cosmetic, dimensional, "
    "material, functional.\n"
    "Severity is an integer 1-5, 5 = safety-critical.\n"
    "Answer as JSON: {\"category\": ..., \"severity\": ...}"
)


def classify(description: str) -> str:
    response = client.messages.create(
        model=MODELS["standard"],
        max_tokens=64,
        temperature=0,          # deterministic labels
        system=SYSTEM,
        messages=[
            {"role": "user", "content": description}
        ],
    )
    return response.content[0].text
