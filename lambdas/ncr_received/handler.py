# lambdas/ncr_received/handler.py
import json
import os

import boto3

sqs = boto3.client("sqs", region_name="us-east-2")
QUEUE = os.environ["TRIAGE_QUEUE_URL"]


def lambda_handler(event, context):
    """SAP posts a new NCR; enqueue it for Build 3."""
    try:
        body = json.loads(event["body"])
        ncr_id = body["ncr_id"]
    except (KeyError, ValueError):
        return {"statusCode": 400,
                "body": '{"error":"bad payload"}'}

    sqs.send_message(
        QueueUrl=QUEUE,
        MessageBody=json.dumps({"ncr_id": ncr_id}),
        MessageGroupId="ncr",
    )
    return {
        "statusCode": 202,
        "body": json.dumps({"queued": ncr_id}),
    }
