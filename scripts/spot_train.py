"""Prepare a one-time Spot launch; --launch explicitly creates a billed resource."""

import argparse
import json
from pathlib import Path


def request(config, bootstrap):
    if not bootstrap.startswith("#!/bin/bash"):
        raise ValueError("reviewed Bash bootstrap required")
    return {
        "ImageId": config["ami_id"],
        "InstanceType": config["instance_type"],
        "MinCount": 1,
        "MaxCount": 1,
        "ClientToken": config["launch_id"],
        "IamInstanceProfile": {
            "Arn": config["instance_profile_arn"]
        },
        "SubnetId": config["subnet_id"],
        "SecurityGroupIds": config["security_group_ids"],
        "UserData": bootstrap,
        "MetadataOptions": {
            "HttpTokens": "required",
            "HttpEndpoint": "enabled",
        },
        "InstanceMarketOptions": {
            "MarketType": "spot",
            "SpotOptions": {
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        },
        "InstanceInitiatedShutdownBehavior": "terminate",
        "TagSpecifications": [
            {
                "ResourceType": "instance",
                "Tags": [
                    {
                        "Key": "Project",
                        "Value": "SupplierQualityBook",
                    },
                    {
                        "Key": "RunId",
                        "Value": config["launch_id"],
                    },
                ],
            }
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--bootstrap", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--launch", action="store_true")
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    payload = request(config, a.bootstrap.read_text())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2) + "\n")
    if a.launch:
        import boto3

        response = boto3.client(
            "ec2", region_name=config["region"]
        ).run_instances(**payload)
        print([i["InstanceId"] for i in response["Instances"]])
    else:
        print("Request saved; no instance launched.")


if __name__ == "__main__":
    main()
