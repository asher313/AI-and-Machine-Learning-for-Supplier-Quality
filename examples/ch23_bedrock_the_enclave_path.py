# Chapter 23 — 23.6 Bedrock — The Enclave Path
# The older path you will still see in existing code.
import json
import os

import boto3

# Legacy ids are version-suffixed, not the Mantle ids
# above; the exact string comes from the console.
LEGACY_ID = "anthropic.claude-sonnet-5-v1:0"

runtime = boto3.client(
    "bedrock-runtime",
    region_name=os.environ["NL_ENCLAVE_REGION"],
)

body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 512,
    "system": "You are an aerospace quality engineer.",
    "messages": [
        {"role": "user", "content": "Classify this NCR"},
    ],
    "temperature": 0,
})

raw = runtime.invoke_model(
    modelId=LEGACY_ID,
    body=body,
    accept="application/json",
    contentType="application/json",
)
result = json.loads(raw["body"].read())
text = result["content"][0]["text"]
