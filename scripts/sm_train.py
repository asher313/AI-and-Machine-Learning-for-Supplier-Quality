"""Prepare a SageMaker Build 1 training request; --submit explicitly starts billing."""

import argparse
import json
from pathlib import Path


def request(config):
    if "@sha256:" not in config["image_uri"]:
        raise ValueError(
            "reviewed immutable training image required"
        )
    return {
        "TrainingJobName": config["job_name"],
        "RoleArn": config["role_arn"],
        "AlgorithmSpecification": {
            "TrainingImage": config["image_uri"],
            "TrainingInputMode": "File",
        },
        "InputDataConfig": [
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": config["input_s3_uri"],
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": config["output_s3_uri"],
            "KmsKeyId": config["kms_key_arn"],
        },
        "ResourceConfig": {
            "InstanceType": config.get(
                "instance_type", "ml.m5.xlarge"
            ),
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
            "VolumeKmsKeyId": config["kms_key_arn"],
        },
        "VpcConfig": {
            "Subnets": config["subnet_ids"],
            "SecurityGroupIds": config["security_group_ids"],
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 3600},
        "EnableNetworkIsolation": True,
        "EnableManagedSpotTraining": False,
        "Tags": [
            {"Key": "Project", "Value": "SupplierQualityBook"}
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    payload = request(config)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2) + "\n")
    if a.submit:
        import boto3

        print(
            boto3.client(
                "sagemaker", region_name=config["region"]
            ).create_training_job(**payload)["TrainingJobArn"]
        )
    else:
        print("Request saved; no training job submitted.")


if __name__ == "__main__":
    main()
