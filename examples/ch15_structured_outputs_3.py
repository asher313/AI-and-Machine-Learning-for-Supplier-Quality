# Chapter 15 teaching listing. Supply the inputs described in the text.
from anthropic import transform_schema
from sqm_ai.llm import MODELS, client, with_retry
from sqm_ai.structured import NCRClassification, SYSTEM

CLASSIFY_TOOL = {
    "name": "record_classification",
    "description": "Return a teaching classification suggestion; do not execute disposition.",
    "strict": True,
    "input_schema": transform_schema(NCRClassification),
}


def classify_via_tool(description):
    # This example uses Haiku with manual thinking disabled.
    response = with_retry(
        client.messages.create,
        model=MODELS["fast"],
        max_tokens=1024,
        system=SYSTEM,
        tools=[CLASSIFY_TOOL],
        tool_choice={
            "type": "tool",
            "name": "record_classification",
            "disable_parallel_tool_use": True,
        },
        messages=[{"role": "user", "content": description}],
    )
    blocks = [b for b in response.content if b.type == "tool_use"]
    if (
        response.stop_reason != "tool_use"
        or len(blocks) != 1
        or blocks[0].name != "record_classification"
    ):
        raise ValueError(
            "expected one complete classification tool block"
        )
    return NCRClassification.model_validate(blocks[0].input)
