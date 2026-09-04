# scripts/sm_train.py
# TODO(book): condensed in Chapter 23 — sample_batch is
# a payload the reader supplies. Complete before
# production use.
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    source_dir="src/sqm_ai/build2",
    role=get_execution_role(),
    framework_version="2.4.0",
    py_version="py312",
    instance_type="ml.g5.xlarge",
    instance_count=1,
    hyperparameters={
        "epochs": 10,
        "batch_size": 64,
        "lr": 1e-3,
    },
    output_path="s3://northlake-sqm-models/defect/",
    use_spot_instances=True,
    max_run=3600,
    max_wait=7200,
    tags=[{"Key": "Project", "Value": "Build2"}],
)

estimator.fit({
    "train": "s3://northlake-sqm-data/defect/train/",
    "validation": "s3://northlake-sqm-data/defect/val/",
})

predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m6i.large",
    endpoint_name="defect-predictor-shadow",
)
print(predictor.predict(sample_batch))
predictor.delete_endpoint()   # endpoints bill hourly
