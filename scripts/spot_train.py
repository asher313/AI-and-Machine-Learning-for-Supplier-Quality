# scripts/spot_train.py  — one-shot training box
import boto3

ec2 = boto3.client("ec2", region_name="us-east-2")

BOOTSTRAP = """#!/bin/bash
set -euo pipefail
pip install uv
aws s3 cp s3://northlake-sqm-data/code/train.tar.gz .
tar xzf train.tar.gz && cd train
uv run python -m sqm_ai.build2.train
aws s3 cp out/model.onnx \\
  s3://northlake-sqm-models/defect/model.onnx
shutdown -h now
"""

ec2.run_instances(
    ImageId="ami-0abcdef0123456789",
    InstanceType="g5.xlarge",
    MinCount=1,
    MaxCount=1,
    IamInstanceProfile={"Name": "sqm-training-role"},
    UserData=BOOTSTRAP,
    InstanceMarketOptions={"MarketType": "spot"},
    TagSpecifications=[{
        "ResourceType": "instance",
        "Tags": [{"Key": "Project", "Value": "Build2"}],
    }],
)
