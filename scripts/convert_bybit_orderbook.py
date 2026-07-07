"""Convert one Bybit quote-saver orderbook ZIP into sampled Top-N parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_mm.data.orderbook_parser import convert_orderbook_zip_to_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one Bybit orderbook ZIP to parquet.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--frequency", default="1s")
    args = parser.parse_args()

    try:
        result = convert_orderbook_zip_to_parquet(
            args.input,
            output_path=args.output,
            symbol=args.symbol,
            depth=args.depth,
            frequency=args.frequency,
        )
    except FileNotFoundError as error:
        parser.exit(status=1, message=f"{error}\n")

    print(f"converted orderbook ZIP to parquet: {result.output_path}")
    print(f"row_count: {result.row_count}")
    print(f"timestamp_range: {result.timestamp_min} -> {result.timestamp_max}")


if __name__ == "__main__":
    main()
