"""Explicit Bedrock adapters for the Chapter 21 text gateway.

Approval is deployment policy, not a property inferred from the SDK or region.
Both constructors require caller-supplied quote/charge functions for actual rates.
"""

from dataclasses import dataclass
import json
from sqm_ai.gateway.router import Endpoint
from sqm_ai.gateway.policy import ENCLAVE_RANK


@dataclass(frozen=True)
class BedrockConfig:
    region: str
    model_id: str
    approved_levels: frozenset[str]
    returned_model_ids: frozenset[str] = frozenset()

    def __post_init__(self):
        object.__setattr__(
            self,
            "approved_levels",
            frozenset(self.approved_levels),
        )
        if (
            not self.region
            or not self.model_id
            or not self.approved_levels
            or not self.approved_levels <= ENCLAVE_RANK.keys()
        ):
            raise ValueError(
                "explicit region, actual model ID and approved classifications required"
            )


def text_response(response):
    result = dict(response)
    result["text"] = "".join(
        b["text"]
        for b in response.get("content", [])
        if b.get("type") == "text"
    )
    return result


def mantle_endpoint(config, *, quote, charge, client=None):
    if client is None:
        from anthropic import AnthropicBedrockMantle

        client = AnthropicBedrockMantle(
            aws_region=config.region, max_retries=0, timeout=60
        )

    def invoke(request):
        return text_response(
            client.messages.create(**request).model_dump(
                mode="json"
            )
        )

    return Endpoint(
        "bedrock-mantle:" + config.region,
        config.model_id,
        config.approved_levels,
        invoke,
        quote,
        charge,
        returned_models=config.returned_model_ids,
    )


def runtime_endpoint(config, *, quote, charge, client=None):
    if client is None:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "bedrock-runtime",
            region_name=config.region,
            config=Config(
                retries={"total_max_attempts": 1}, read_timeout=60
            ),
        )

    def invoke(request):
        body = {k: v for k, v in request.items() if k != "model"}
        body["anthropic_version"] = "bedrock-2023-05-31"
        raw = client.invoke_model(
            modelId=config.model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        try:
            return text_response(json.loads(raw["body"].read()))
        finally:
            raw["body"].close()

    return Endpoint(
        "bedrock-runtime:" + config.region,
        config.model_id,
        config.approved_levels,
        invoke,
        quote,
        charge,
        returned_models=config.returned_model_ids,
    )
