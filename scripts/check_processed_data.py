"""Run quality checks on a processed Bybit parquet file."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_mm.data.quality import check_processed_data, print_quality_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check processed Bybit parquet data.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", choices=["trades", "orderbook"], required=True)
    args = parser.parse_args()

    try:
        report = check_processed_data(args.input, dataset=args.dataset)
    except FileNotFoundError as error:
        parser.exit(status=1, message=f"{error}\n")
    print_quality_report(report)


if __name__ == "__main__":
    main()
