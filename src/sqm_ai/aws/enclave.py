# src/sqm_ai/aws/enclave.py
from __future__ import annotations

import os

from anthropic import AnthropicBedrockMantle

from sqm_ai.llm import MODELS

enclave = AnthropicBedrockMantle(
    aws_region=os.environ["NL_ENCLAVE_REGION"],
)


def classify(description: str) -> str:
    """Same call shape as Chapter 15, inside the enclave."""
    response = enclave.messages.create(
        model="anthropic." + MODELS["standard"],
        max_tokens=512,
        system=(
            "You are an aerospace supplier-quality "
            "engineer. Answer only from the text given."
        ),
        messages=[
            {"role": "user", "content": description},
        ],
    )
    return response.content[0].text
