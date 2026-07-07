"""Plan Bybit historical market-data downloads or validate local files."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_mm.data.bybit_downloader import (
    build_download_plan,
    download_plan,
    load_bybit_config,
    print_dry_run,
    validate_local_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bybit data ingestion scaffold.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_bybit.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Plan only; no network calls.")
    parser.add_argument("--local-file", type=Path)
    parser.add_argument("--dataset", choices=["trades", "orderbook"])
    parser.add_argument("--symbol")
    parser.add_argument("--start-date", required=False)
    parser.add_argument("--end-date", required=False)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Download files. Default is dry-run.",
    )
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--timeout", type=float)
    args = parser.parse_args()

    if args.execute and args.dry_run:
        parser.error("--execute and --dry-run cannot be used together")

    config = load_bybit_config(args.config)

    if args.local_file is not None:
        dataset = args.dataset or "trades"
        validate_local_csv(args.local_file, dataset=dataset)
        print(f"validated {dataset} file: {args.local_file}")
        return

    plan = build_download_plan(
        config,
        dataset=args.dataset,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        max_files=args.max_files,
    )
    if not args.execute:
        print_dry_run(plan)
        return

    timeout = args.timeout or float(config.get("timeout_seconds", 30))
    results = download_plan(plan, timeout=timeout)
    print("Bybit download results")
    for result in results:
        item = result.plan
        print(
            f"- {item.dataset} {item.symbol} {item.date}: "
            f"{result.status} bytes={result.bytes_written} raw={item.raw_path}"
        )


if __name__ == "__main__":
    main()
