"""Local release rehearsal; no cloud registration or running-service deployment."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register(model, gate, code_sha, run_id):
    model, gate = Path(model), Path(gate)
    result = json.loads(gate.read_text())
    if result.get("passed") is not True or result.get(
        "metrics_sha256"
    ) != digest(model / "metrics.json"):
        raise ValueError(
            "passing gate for these exact metrics required"
        )
    if not code_sha or not run_id:
        raise ValueError(
            "code revision and unique run ID required"
        )
    metadata = json.loads((model / "metadata.json").read_text())
    for name, expected in metadata["artifact_sha256"].items():
        if (
            Path(name).name != name
            or digest(model / name) != expected
        ):
            raise ValueError("model artifact hash mismatch")
    files = {
        p.name: digest(p)
        for p in sorted(model.iterdir())
        if p.is_file()
    }
    required = {
        "risk_regressor.joblib",
        "risk_classifier_calibrated.joblib",
        "metadata.json",
        "metrics.json",
        "feature_list.json",
        "threshold.json",
    }
    if not required <= files.keys():
        raise ValueError("complete fitted model package required")
    return {
        "kind": "synthetic-release-rehearsal",
        "code_sha": code_sha,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": metadata["model_version"],
        "training_data_sha256": metadata["training_data_sha256"],
        "gate": result,
        "artifact_sha256": files,
        "intended_use": "Teaching demonstration; no production approval",
    }


def rehearse(release, model, target):
    if release.get(
        "kind"
    ) != "synthetic-release-rehearsal" or target not in {
        "staging",
        "production",
    }:
        raise ValueError(
            "explicit rehearsal release and target required"
        )
    for name, expected in release["artifact_sha256"].items():
        if (
            Path(name).name != name
            or digest(Path(model) / name) != expected
        ):
            raise ValueError("release artifact hash mismatch")
    return {
        "mode": "local-rehearsal",
        "target": target,
        "run_id": release["run_id"],
        "model_version": release["model_version"],
        "service_deployed": False,
        "checked_artifacts": len(release["artifact_sha256"]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("register")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--gate", type=Path, required=True)
    r.add_argument("--code-sha", required=True)
    r.add_argument("--run-id", required=True)
    r.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("deploy")
    d.add_argument("--model", type=Path, required=True)
    d.add_argument("--release", type=Path, required=True)
    d.add_argument(
        "--target",
        choices=["staging", "production"],
        required=True,
    )
    d.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = (
        register(
            args.model, args.gate, args.code_sha, args.run_id
        )
        if args.command == "register"
        else rehearse(
            json.loads(args.release.read_text()),
            args.model,
            args.target,
        )
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
