# Chapter 15 teaching listing. Supply the inputs described in the text.
from anthropic import transform_schema
from sqm_ai.llm import text_response, with_retry
from sqm_ai.structured import NCRClassification, SYSTEM

response = with_retry(
    client.messages.create,
    model=MODELS["standard"], max_tokens=1024, system=SYSTEM,
    messages=[{"role": "user", "content": description}],
    output_config={"effort": "low", "format": {
        "type": "json_schema", "schema": transform_schema(NCRClassification),
    }},
)
result = NCRClassification.model_validate_json(text_response(response))
