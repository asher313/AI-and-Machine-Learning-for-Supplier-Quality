"""Validated FIFO intake; authentication is enforced by the configured API front door."""

import base64
import hashlib
import json
import os
import re


def enqueue(event, *, client, queue_url):
    # Client-provided JSON fields cannot supply this identity; bind an API authorizer.
    auth = event.get("requestContext", {}).get("authorizer", {})
    principal = auth.get("principalId") or auth.get(
        "jwt", {}
    ).get("claims", {}).get("sub")
    if not principal:
        return {
            "statusCode": 401,
            "body": '{"error":"unauthorized"}',
        }
    if not queue_url.endswith(".fifo"):
        raise ValueError("FIFO queue required")
    try:
        raw = event["body"]
        if event.get("isBase64Encoded"):
            raw = base64.b64decode(raw, validate=True).decode(
                "utf-8"
            )
        if not isinstance(raw, str) or len(raw) > 10000:
            raise ValueError("invalid payload size")
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("object required")
        ncr_id = body["ncr_id"]
        event_id = body["event_id"]
        if not isinstance(ncr_id, str) or not re.fullmatch(
            r"NCR-\d{4}-\d{4,}", ncr_id
        ):
            raise ValueError("invalid NCR ID")
        if not isinstance(event_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_.:-]{1,128}", event_id
        ):
            raise ValueError("stable source event ID required")
    except (KeyError, TypeError, ValueError, UnicodeError):
        return {
            "statusCode": 400,
            "body": '{"error":"bad payload"}',
        }
    # Trusted source retries must preserve event_id. Consumer also needs durable idempotency.
    dedup = hashlib.sha256(
        f"{principal}:{event_id}".encode()
    ).hexdigest()
    client.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(
            {"ncr_id": ncr_id, "event_id": event_id}
        ),
        MessageGroupId=ncr_id,
        MessageDeduplicationId=dedup,
    )
    return {
        "statusCode": 202,
        "body": json.dumps({"queued": ncr_id}),
    }


def lambda_handler(event, context):
    import boto3

    return enqueue(
        event,
        client=boto3.client(
            "sqs", region_name=os.environ["AWS_REGION"]
        ),
        queue_url=os.environ["TRIAGE_QUEUE_URL"],
    )
