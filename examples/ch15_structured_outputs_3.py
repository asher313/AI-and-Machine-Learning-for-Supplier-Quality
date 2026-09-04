# Chapter 15 — 15.3 Structured Outputs
schema = NCRClassification.model_json_schema()
# strict mode requires this key:
schema["additionalProperties"] = False

CLASSIFY_TOOL = {
    "name": "record_classification",
    "description": "Record the classification of one NCR.",
    "strict": True,
    "input_schema": schema,
}


def classify_via_tool(description: str) -> NCRClassification:
    response = client.messages.create(
        model=MODELS["standard"],
        max_tokens=512,
        system=SYSTEM,
        tools=[CLASSIFY_TOOL],
        tool_choice={
            "type": "tool",
            "name": "record_classification",
        },
        messages=[{"role": "user", "content": description}],
    )
    block = next(
        b for b in response.content if b.type == "tool_use"
    )
    return NCRClassification.model_validate(block.input)
