"""Plan Bybit historical market-data downloads or validate local files."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_mm.data.bybit_downloader import (
    build_download_plan,
    load_bybit_config,
    print_dry_run,
    validate_local_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bybit data ingestion scaffold.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_bybit.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-file", type=Path)
    parser.add_argument("--dataset", choices=["trades", "orderbook"], default="trades")
    args = parser.parse_args()

    config = load_bybit_config(args.config)

    if args.local_file is not None:
        validate_local_csv(args.local_file, dataset=args.dataset)
        print(f"validated {args.dataset} file: {args.local_file}")
        return

    plan = build_download_plan(config)
    if args.dry_run:
        print_dry_run(plan)
        return

    raise NotImplementedError(
        "Network download is not implemented yet. Use --dry-run or --local-file for now."
    )


if __name__ == "__main__":
    main()
