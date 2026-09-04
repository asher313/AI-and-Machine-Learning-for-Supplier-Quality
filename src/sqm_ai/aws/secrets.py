# src/sqm_ai/aws/secrets.py
import json
from functools import lru_cache

import boto3

sm = boto3.client("secretsmanager",
                  region_name="us-east-2")


@lru_cache(maxsize=8)
def get_secret(name: str) -> dict:
    """Fetch and cache one secret for this process."""
    resp = sm.get_secret_value(SecretId=name)
    return json.loads(resp["SecretString"])


creds = get_secret("sqm-ai/prod/hana")
# creds["user"], creds["password"] -> Chapter 5.4
