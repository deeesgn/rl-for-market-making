"""Convert validated local Bybit CSV data to normalized parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_mm.data.converter import convert_csv_to_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert local Bybit CSV to parquet.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", choices=["trades", "orderbook"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        output_path = convert_csv_to_parquet(
            args.input,
            dataset=args.dataset,
            output_path=args.output,
        )
    except FileNotFoundError as error:
        message = f"{error}\n"
        if "data/raw/bybit/verify" in args.input.as_posix():
            message += "Run `make verify-bybit-archive-sample` first.\n"
        parser.exit(status=1, message=message)

    print(f"converted {args.dataset} CSV to parquet: {output_path}")


if __name__ == "__main__":
    main()
