# src/sqm_ai/aws/metrics.py
from datetime import datetime, timezone

import boto3

cw = boto3.client("cloudwatch", region_name="us-east-2")
NS = "SQM/NCRClassifier"


def record_confidence(value: float, model: str) -> None:
    cw.put_metric_data(
        Namespace=NS,
        MetricData=[{
            "MetricName": "ClassificationConfidence",
            "Value": value,
            "Unit": "None",
            "Dimensions": [
                {"Name": "Model", "Value": model},
                {"Name": "Environment", "Value": "prod"},
            ],
            "Timestamp": datetime.now(timezone.utc),
        }],
    )


def create_low_confidence_alarm(topic_arn: str) -> None:
    cw.put_metric_alarm(
        AlarmName="NCRLowConfidence",
        Namespace=NS,
        MetricName="ClassificationConfidence",
        Statistic="Average",
        Period=300,
        EvaluationPeriods=3,
        Threshold=0.70,
        ComparisonOperator="LessThanThreshold",
        AlarmActions=[topic_arn],
    )
