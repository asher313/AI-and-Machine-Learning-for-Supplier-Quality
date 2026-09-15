"""AWS protocol checks use Botocore Stubber; no cloud resources or billable calls."""

from io import BytesIO
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import boto3
from botocore.response import StreamingBody
from botocore.stub import Stubber, ANY
from botocore.exceptions import ClientError
import pandas as pd
import pytest
from sqm_ai.aws.storage import list_keys, read_parquet, share_link
from sqm_ai.aws.secrets import SecretCache
from sqm_ai.aws.metrics import (
    record_confidence,
    create_low_confidence_alarm,
    dimensions,
    NS,
)
from sqm_ai.aws.enclave import (
    BedrockConfig,
    mantle_endpoint,
    runtime_endpoint,
)


def client(service):
    return boto3.client(
        service,
        region_name="us-east-2",
        aws_access_key_id="fake",
        aws_secret_access_key="fake",
    )


def test_s3_pagination_and_seekable_parquet_download():
    s3 = client("s3")
    output = BytesIO()
    pd.DataFrame({"x": [1, 2]}).to_parquet(output, index=False)
    data = output.getvalue()
    body = StreamingBody(BytesIO(data), len(data))
    with Stubber(s3) as stub:
        stub.add_response(
            "list_objects_v2",
            {
                "IsTruncated": True,
                "NextContinuationToken": "page2",
                "Contents": [{"Key": "features/a"}],
            },
            {"Bucket": "fixture", "Prefix": "features/"},
        )
        stub.add_response(
            "list_objects_v2",
            {
                "IsTruncated": False,
                "Contents": [{"Key": "features/b"}],
            },
            {
                "Bucket": "fixture",
                "Prefix": "features/",
                "ContinuationToken": "page2",
            },
        )
        stub.add_response(
            "get_object",
            {"Body": body, "ContentLength": len(data)},
            {"Bucket": "fixture", "Key": "features/a"},
        )
        assert list(
            list_keys("fixture", "features/", client=s3)
        ) == ["features/a", "features/b"]
        assert read_parquet(
            "fixture", "features/a", client=s3
        ).x.tolist() == [1, 2]
        assert body._raw_stream.closed
        stub.assert_no_pending_responses()


def test_download_limit_closes_body_and_share_requires_approval():
    s3 = client("s3")
    body = StreamingBody(BytesIO(b"too large"), 9)
    with Stubber(s3) as stub:
        stub.add_response(
            "get_object",
            {"Body": body, "ContentLength": 9},
            {"Bucket": "fixture", "Key": "k"},
        )
        with pytest.raises(ValueError):
            read_parquet("fixture", "k", client=s3, max_bytes=2)
        assert body._raw_stream.closed
        with pytest.raises(ValueError):
            share_link(
                "fixture",
                "k",
                client=s3,
                classification="controlled",
                allow_external_share=True,
            )
        with pytest.raises(ValueError):
            share_link("fixture", "k", client=s3)


def test_secret_rotation_refresh_and_no_stale_fallback():
    sm = client("secretsmanager")
    now = [0]
    cache = SecretCache(sm, ttl_seconds=5, clock=lambda: now[0])
    expected = {
        "SecretId": "synthetic-db",
        "VersionStage": "AWSCURRENT",
    }
    with Stubber(sm) as stub:
        stub.add_response(
            "get_secret_value",
            {"SecretString": '{"password":"old-fixture"}'},
            expected,
        )
        stub.add_response(
            "get_secret_value",
            {"SecretString": '{"password":"new-fixture"}'},
            expected,
        )
        stub.add_client_error(
            "get_secret_value",
            "AccessDeniedException",
            expected_params=expected,
        )
        first = cache.get_secret("synthetic-db")
        first["password"] = "caller mutation"
        assert (
            cache.get_secret("synthetic-db")["password"]
            == "old-fixture"
        )
        now[0] = 6
        assert (
            cache.get_secret("synthetic-db")["password"]
            == "new-fixture"
        )
        now[0] = 12
        with pytest.raises(ClientError):
            cache.get_secret("synthetic-db")
        assert not cache.values
        stub.assert_no_pending_responses()


def test_metric_alarm_selects_exact_dimensions():
    cw = client("cloudwatch")
    dims = dimensions("synthetic-model", "test")
    with Stubber(cw) as stub:
        stub.add_response(
            "put_metric_data",
            {},
            {
                "Namespace": NS,
                "MetricData": [
                    {
                        "MetricName": "ClassificationConfidence",
                        "Value": 0.6,
                        "Unit": "None",
                        "Dimensions": dims,
                        "Timestamp": ANY,
                    }
                ],
            },
        )
        stub.add_response(
            "put_metric_alarm",
            {},
            {
                "AlarmName": "NCRLowConfidence-test-synthetic-model",
                "Namespace": NS,
                "MetricName": "ClassificationConfidence",
                "Dimensions": dims,
                "Statistic": "Average",
                "Period": 300,
                "EvaluationPeriods": 3,
                "DatapointsToAlarm": 3,
                "Threshold": 0.7,
                "ComparisonOperator": "LessThanThreshold",
                "TreatMissingData": "missing",
                "AlarmActions": [
                    "arn:aws:sns:us-east-2:123456789012:fixture"
                ],
            },
        )
        record_confidence(
            0.6, "synthetic-model", client=cw, environment="test"
        )
        create_low_confidence_alarm(
            "arn:aws:sns:us-east-2:123456789012:fixture",
            "synthetic-model",
            client=cw,
            environment="test",
        )
        with pytest.raises(ValueError):
            record_confidence(
                float("nan"), "synthetic-model", client=cw
            )
        stub.assert_no_pending_responses()


def test_bedrock_transports_normalize_into_gateway_endpoint():
    config = BedrockConfig(
        "us-east-2",
        "operator-selected-fixture-model",
        frozenset({"open"}),
    )
    request = {
        "model": config.model_id,
        "max_tokens": 32,
        "system": "Synthetic only",
        "messages": [{"role": "user", "content": "Test"}],
    }
    result = {
        "model": config.model_id,
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "Fixture answer"}],
        "usage": {"input_tokens": 4, "output_tokens": 3},
    }
    seen = []

    def create(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(model_dump=lambda **kw: result)

    ep = mantle_endpoint(
        config,
        quote=lambda r: 0.02,
        charge=lambda r: 0.01,
        client=SimpleNamespace(
            messages=SimpleNamespace(create=create)
        ),
    )
    assert ep.invoke(request)[
        "text"
    ] == "Fixture answer" and seen == [request]
    runtime = client("bedrock-runtime")
    raw = json.dumps(result).encode()
    body = StreamingBody(BytesIO(raw), len(raw))
    expected_body = {
        k: v for k, v in request.items() if k != "model"
    }
    expected_body["anthropic_version"] = "bedrock-2023-05-31"
    with Stubber(runtime) as stub:
        stub.add_response(
            "invoke_model",
            {"body": body, "contentType": "application/json"},
            {
                "modelId": config.model_id,
                "body": json.dumps(expected_body),
                "accept": "application/json",
                "contentType": "application/json",
            },
        )
        ep = runtime_endpoint(
            config,
            quote=lambda r: 0.02,
            charge=lambda r: 0.01,
            client=runtime,
        )
        assert (
            ep.invoke(request)["text"] == "Fixture answer"
            and body._raw_stream.closed
        )
        stub.assert_no_pending_responses()


def test_fifo_intake_validates_and_reuses_stable_source_event_id():
    path = (
        Path(__file__).parents[1]
        / "lambdas/ncr_received/handler.py"
    )
    spec = importlib.util.spec_from_file_location("intake", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sent = []
    sqs = SimpleNamespace(
        send_message=lambda **kw: sent.append(kw)
    )
    queue = "https://sqs.us-east-2.amazonaws.com/123456789012/fixture.fifo"
    event = {
        "requestContext": {
            "authorizer": {"principalId": "trusted-fixture"}
        },
        "body": json.dumps(
            {"ncr_id": "NCR-2026-0042", "event_id": "created-42"}
        ),
    }
    assert (
        module.enqueue(event, client=sqs, queue_url=queue)[
            "statusCode"
        ]
        == 202
    )
    assert (
        module.enqueue(event, client=sqs, queue_url=queue)[
            "statusCode"
        ]
        == 202
    )
    assert (
        sent[0]["MessageDeduplicationId"]
        == sent[1]["MessageDeduplicationId"]
        and sent[0]["MessageGroupId"] == "NCR-2026-0042"
    )
    for body in [
        "null",
        "[]",
        '{"ncr_id":"wrong","event_id":"x"}',
        '{"ncr_id":"NCR-2026-0042"}',
    ]:
        assert (
            module.enqueue(
                dict(event, body=body),
                client=sqs,
                queue_url=queue,
            )["statusCode"]
            == 400
        )
    assert (
        module.enqueue(
            {"body": event["body"]}, client=sqs, queue_url=queue
        )["statusCode"]
        == 401
    )
    assert len(sent) == 2


def test_generated_cloud_requests_match_sdk_shapes_without_submission(
    tmp_path,
):
    from botocore.validate import validate_parameters

    root = Path(__file__).parents[1]

    def load(name):
        spec = importlib.util.spec_from_file_location(
            name, root / "scripts" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    load("generate_cloud_data").generate(tmp_path)
    sm = load("sm_train").request(
        json.loads((tmp_path / "sagemaker.json").read_text())
    )
    spot = load("spot_train").request(
        json.loads((tmp_path / "spot.json").read_text()),
        (tmp_path / "bootstrap.sh").read_text(),
    )
    task = load("render_template").render(
        json.loads(
            (
                root / "examples/ch23_ecs_task_definition.json"
            ).read_text()
        ),
        json.loads((tmp_path / "ecs-values.json").read_text()),
    )
    for service, operation, payload in [
        ("sagemaker", "CreateTrainingJob", sm),
        ("ec2", "RunInstances", spot),
        ("ecs", "RegisterTaskDefinition", task),
    ]:
        api = client(service)
        validate_parameters(
            payload,
            api.meta.service_model.operation_model(
                operation
            ).input_shape,
        )
    assert (
        sm["EnableNetworkIsolation"]
        and not sm["EnableManagedSpotTraining"]
    )
    assert (
        spot["InstanceInitiatedShutdownBehavior"] == "terminate"
        and spot["MetadataOptions"]["HttpTokens"] == "required"
    )
    assert task["executionRoleArn"] != task[
        "taskRoleArn"
    ] and "$" not in json.dumps(task)
