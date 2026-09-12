"""S3 operations with injected clients and explicit storage/sharing policy."""

from io import BytesIO
import pandas as pd


def upload(path, bucket, key, *, client, kms_key_arn):
    if (
        not kms_key_arn.startswith("arn:")
        or ":kms:" not in kms_key_arn
    ):
        raise ValueError("approved KMS key ARN required")
    client.upload_file(
        Filename=str(path),
        Bucket=bucket,
        Key=key,
        ExtraArgs={
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": kms_key_arn,
            "Metadata": {"produced_by": "sqm-pipeline"},
        },
    )


def list_keys(bucket, prefix, *, client):
    """Yield all pages without retaining the whole key list in memory."""
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    ):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def read_parquet(
    bucket, key, *, client, max_bytes=64 * 1024 * 1024
):
    """Download a bounded object into seekable memory; decoded data may be larger."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("positive byte limit required")
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    try:
        if response.get("ContentLength", 0) > max_bytes:
            raise ValueError(
                "object exceeds memory download limit"
            )
        raw = body.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError(
                "object exceeds memory download limit"
            )
    finally:
        body.close()
    return pd.read_parquet(BytesIO(raw))


def share_link(
    bucket,
    key,
    *,
    client,
    classification="unknown",
    allow_external_share=False,
    seconds=3600,
):
    """Fictional policy permits only explicitly approved open-data sharing."""
    if classification != "open" or not allow_external_share:
        raise ValueError("external sharing is not authorized")
    if type(seconds) is not int or not 1 <= seconds <= 3600:
        raise ValueError("expiry must be 1..3600 seconds")
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=seconds,
    )
