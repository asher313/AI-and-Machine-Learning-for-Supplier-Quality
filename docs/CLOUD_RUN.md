# Chapter 23: offline cloud contract exercises

The cloud functions accept injected clients. Tests use Botocore Stubber or scripted SDK objects; they create no AWS resources and make no model API calls. Configuration fixtures contain fictional account/resource identifiers, not credentials.

```bash
uv sync --locked --extra ml
uv run --no-sync pytest tests/test_aws_contracts.py
uv run --no-sync python scripts/generate_cloud_data.py --out work/cloud
uv run --no-sync python scripts/sm_train.py --config work/cloud/sagemaker.json --out work/cloud/training-request.json
uv run --no-sync python scripts/spot_train.py --config work/cloud/spot.json --bootstrap work/cloud/bootstrap.sh --out work/cloud/spot-request.json
uv run --no-sync python scripts/render_template.py --template examples/ch23_ecs_task_definition.json --values work/cloud/ecs-values.json --out work/cloud/ecs-task.json
```

Seven local tests cover paginated S3 reads, bounded seekable Parquet loading and body cleanup, sharing authorization, secret TTL/rotation, exact metric/alarm dimensions, both Bedrock text transports, FIFO intake validation/idempotency fields, and generated request shapes for EC2, SageMaker and ECS. Shape validation does not confirm resource existence, IAM authorization, capacity, legal eligibility or network reachability.

Training and launch scripts only save request JSON by default. Their explicit `--submit` and `--launch` flags call AWS and can incur charges. Do not use the fictional fixture resources with those flags. A real submission requires operator-reviewed identifiers, roles, key/bucket policies, region/partition and service availability, network controls, budgets and cleanup. Neither account separation nor a Bedrock endpoint automatically authorizes controlled-data processing.

`docker/Dockerfile.training` and `scripts/sagemaker_entrypoint.py` provide a custom Build 1 training image contract. Build from the repository root. File-mode input is `/opt/ml/input/data/train/supplier_month.parquet`; output is `/opt/ml/model/build1`. The request uses network isolation and on-demand training; dependencies must already be in the image. Managed Spot recovery is not claimed because the trainer has no complete checkpoint/resume path. The image was not built or submitted to SageMaker during this review. No endpoint is automatically deployed. Hosting needs a separate validated inference implementation.

The Spot bootstrap fixture assumes a prepared AMI containing the checkout, Python environment and synthetic data at `/opt/supplier-book`. It is a small CPU exercise; it neither downloads arbitrary code nor uploads results. A real job must preserve results at an approved immutable destination before termination and implement recovery where needed. An exit trap and requested shutdown are not a substitute for independent time limits, resource inventory and cleanup verification. Storage and other resources can continue billing after an instance stops or terminates.

The ECS and Kubernetes examples describe an independently implemented NCR API. This repository's training image is not that API. The ECS JSON renderer substitutes configuration into parsed string values and fails on missing values; AWS does not expand `$VARIABLE` strings in raw JSON. The execution role handles startup image/log/secret operations; the task role grants application permissions. Rotated injected secrets require replacement tasks or a runtime refresh design. The examples do not provision a service, ingress, load balancer, API authorization, network policy or managed key lifecycle.

The FIFO Lambda handler expects trusted API authorizer context and a stable source `event_id`. Deduplication is bounded by SQS's window; consumers must enforce durable idempotency and source-record authorization. Standard queue configuration is not supported by this particular handler. Confidence telemetry is not proof of calibrated accuracy; monitor outcomes and availability separately.

For runtime AWS clients, use approved workload roles/federation and the SDK credential chain. Never publish keys, secret values, presigned URLs or sensitive request content in logs. The local `SecretCache` refreshes current JSON secrets by TTL/explicit refresh; application connection pools may also require renewal. The Bedrock factories integrate with the Chapter 21 text gateway using explicit approved model IDs and account-specific quote/charge functions. Other provider interfaces still require their own integration and tests.
