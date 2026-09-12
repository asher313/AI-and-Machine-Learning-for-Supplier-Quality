"""Generate the textbook's local, explicitly synthetic datasets."""

import argparse
from pathlib import Path

from sqm_ai.cnc.synthetic import generate as cnc_data
from sqm_ai.synthetic import generate as supplier_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data")
    )
    parser.add_argument("--skip-cnc", action="store_true")
    parser.add_argument("--cnc-cycles", type=int, default=931240)
    parser.add_argument("--raw-cycles", type=int, default=40000)
    args = parser.parse_args()
    print(
        "Generating NCR, supplier-month and Cobalt lot data...",
        flush=True,
    )
    supplier_data(args.output)
    if not args.skip_cnc:
        print(
            "Generating CNC features and pilot windows...",
            flush=True,
        )
        cnc_data(
            args.output / "cnc", args.cnc_cycles, args.raw_cycles
        )
    print(f"Generated synthetic data in {args.output.resolve()}")
    print(
        "Run python -m sqm_ai.verify_synthetic --data "
        f'"{args.output}"'
        + (
            ""
            if args.skip_cnc
            else f' --cnc "{args.output / "cnc"}"'
        )
    )


if __name__ == "__main__":
    main()
