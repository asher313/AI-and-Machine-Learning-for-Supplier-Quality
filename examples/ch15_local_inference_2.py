# Chapter 15 teaching listing. Supply the inputs described in the text.
import httpx

# SYSTEM and NCRClassification: imports as in §15.3.
VLLM = "http://vllm.northlake.internal:8000/v1/chat/completions"

payload = {
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": description},
    ],
    "temperature": 0,
    "max_tokens": 400,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "ncr",
            "schema": NCRClassification.model_json_schema(),
        },
    },
}
r = httpx.post(VLLM, json=payload, timeout=60.0)
r.raise_for_status()
text = r.json()["choices"][0]["message"]["content"]
result = NCRClassification.model_validate_json(text)
