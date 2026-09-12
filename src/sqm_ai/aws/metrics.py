"""Publish and alarm on the same dimensioned custom metric."""

from datetime import datetime, timezone
import math

NS = "SQM/NCRClassifier"


def dimensions(model, environment):
    if (
        not isinstance(model, str)
        or not isinstance(environment, str)
        or not 1 <= len(model) <= 128
        or not 1 <= len(environment) <= 64
    ):
        raise ValueError(
            "bounded model and environment labels required"
        )
    return [
        {"Name": "Model", "Value": model},
        {"Name": "Environment", "Value": environment},
    ]


def record_confidence(
    value, model, *, client, environment="prod"
):
    if (
        isinstance(value, bool)
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("finite confidence in [0,1] required")
    client.put_metric_data(
        Namespace=NS,
        MetricData=[
            {
                "MetricName": "ClassificationConfidence",
                "Value": value,
                "Unit": "None",
                "Dimensions": dimensions(model, environment),
                "Timestamp": datetime.now(timezone.utc),
            }
        ],
    )


def create_low_confidence_alarm(
    topic_arn, model, *, client, environment="prod"
):
    client.put_metric_alarm(
        AlarmName=f"NCRLowConfidence-{environment}-{model}",
        Namespace=NS,
        MetricName="ClassificationConfidence",
        Dimensions=dimensions(model, environment),
        Statistic="Average",
        Period=300,
        EvaluationPeriods=3,
        DatapointsToAlarm=3,
        Threshold=0.70,
        ComparisonOperator="LessThanThreshold",
        TreatMissingData="missing",
        AlarmActions=[topic_arn],
    )
