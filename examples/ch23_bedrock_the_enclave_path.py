# Runtime factory excerpt; same gateway Endpoint contract.
import json
from sqm_ai.gateway.router import Endpoint
from sqm_ai.aws.enclave import text_response

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
    )
