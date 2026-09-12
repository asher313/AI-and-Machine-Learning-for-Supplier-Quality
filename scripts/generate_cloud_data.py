"""Generate fictional cloud configuration fixtures, never real credentials."""

import argparse
import json
from pathlib import Path


def generate(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    region = "us-east-2"
    account = "123456789012"
    kms = f"arn:aws:kms:{region}:{account}:key/00000000-0000-0000-0000-000000000001"
    sm = {
        "region": region,
        "job_name": "supplier-book-synthetic-plan",
        "role_arn": f"arn:aws:iam::{account}:role/fictional-training",
        "image_uri": f"{account}.dkr.ecr.{region}.amazonaws.com/fictional-training@sha256:"
        + 64 * "0",
        "input_s3_uri": "s3://fictional-sqm-input/train/",
        "output_s3_uri": "s3://fictional-sqm-output/build1/",
        "kms_key_arn": kms,
        "subnet_ids": ["subnet-00000000000000001"],
        "security_group_ids": ["sg-00000000000000001"],
    }
    spot = {
        "region": region,
        "ami_id": "ami-00000000000000001",
        "instance_type": "m5.xlarge",
        "launch_id": "synthetic-plan-001",
        "instance_profile_arn": f"arn:aws:iam::{account}:instance-profile/fictional-training",
        "subnet_id": "subnet-00000000000000001",
        "security_group_ids": ["sg-00000000000000001"],
    }
    for name, value in [
        ("sagemaker.json", sm),
        ("spot.json", spot),
    ]:
        (out / name).write_text(
            json.dumps(value, indent=2) + "\n"
        )
    ecs = {
        "TASK_ROLE_ARN": f"arn:aws:iam::{account}:role/fictional-app",
        "EXECUTION_ROLE_ARN": f"arn:aws:iam::{account}:role/fictional-ecs-execution",
        "APP_IMAGE_DIGEST": f"{account}.dkr.ecr.{region}.amazonaws.com/fictional-app@sha256:"
        + 64 * "0",
        "ANTHROPIC_SECRET_ARN": f"arn:aws:secretsmanager:{region}:{account}:secret:fictional-api-abcdef",
        "AWS_REGION": region,
    }
    (out / "ecs-values.json").write_text(
        json.dumps(ecs, indent=2) + "\n"
    )
    # A prepared AMI must contain this checkout, environment, input and upload script.
    # This fixture runs a bounded synthetic CPU exercise; no resume guarantee.
    bootstrap = """#!/bin/bash
set -euo pipefail
trap 'shutdown -h now' EXIT
cd /opt/supplier-book
.venv/bin/python -m sqm_ai.build1.train --input data/supplier_month.parquet --output artifacts/model --trees 30
# Upload to an approved unique run location before shutdown in a real job.
"""
    (out / "bootstrap.sh").write_text(bootstrap)
    (out / "README.txt").write_text(
        "All accounts, resources and digests in these files are fictional. Use only dry-run payload generation and mocked validation. Replace and independently review every resource before any explicit cloud submission.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    generate(parser.parse_args().out)
