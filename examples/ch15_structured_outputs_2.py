# Chapter 15 — 15.3 Structured Outputs
response = client.messages.create(
    model=MODELS["standard"],
    max_tokens=512,
    system=SYSTEM,
    messages=[{"role": "user", "content": description}],
    output_config={
        "effort": "low",
        "format": {
            "type": "json_schema",
            "schema": NCRClassification.model_json_schema(),
        },
    },
)
text = next(b.text for b in response.content if b.type == "text")
result = NCRClassification.model_validate_json(text)
