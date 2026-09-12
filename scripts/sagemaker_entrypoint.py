"""Custom-image contract: File-mode train channel in; Build 1 model package out."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", default="train", choices=["train"]
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "/opt/ml/input/data/train/supplier_month.parquet"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/opt/ml/model/build1"),
    )
    parser.add_argument("--trees", type=int)
    args = parser.parse_args()
    cmd = [
        sys.executable,
        "-m",
        "sqm_ai.build1.train",
        "--input",
        str(args.input),
        "--output",
        str(args.output),
    ]
    if args.trees is not None:
        cmd += ["--trees", str(args.trees)]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
