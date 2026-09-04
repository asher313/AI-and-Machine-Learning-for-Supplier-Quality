# src/sqm_ai/aws/storage.py
from __future__ import annotations

import boto3
import pandas as pd

s3 = boto3.client("s3", region_name="us-east-2")
KMS_KEY = "alias/northlake-sqm"


def upload(path: str, bucket: str, key: str) -> None:
    """Upload one file, always encrypted."""
    s3.upload_file(
        Filename=path,
        Bucket=bucket,
        Key=key,
        ExtraArgs={
            "ServerSideEncryption": "aws:kms",
            "SSEKMSKeyId": KMS_KEY,
            "Metadata": {"produced_by": "sqm-pipeline"},
        },
    )


def list_keys(bucket: str, prefix: str) -> list[str]:
    """Every key under a prefix; buckets can be huge."""
    pages = s3.get_paginator("list_objects_v2")
    out: list[str] = []
    for page in pages.paginate(Bucket=bucket,
                               Prefix=prefix):
        for obj in page.get("Contents", []):
            out.append(obj["Key"])
    return out


def read_parquet(bucket: str, key: str) -> pd.DataFrame:
    """Stream an object; never land it on disk."""
    resp = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(resp["Body"])


def share_link(bucket: str, key: str,
               seconds: int = 3600) -> str:
    """Time-limited URL. Never for controlled data."""
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=seconds,
    )
