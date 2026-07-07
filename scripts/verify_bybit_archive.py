"""Verify that one Bybit archive can be downloaded and read."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_mm.data.archive import inspect_csv_archive, inspect_jsonl_data_archive
from rl_mm.data.bybit_downloader import build_download_plan, download_file, load_bybit_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify one Bybit historical archive.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_bybit.yaml"))
    parser.add_argument("--dataset", choices=["trades", "orderbook"], required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template-name")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = load_bybit_config(args.config)
    verification_config = {
        **config,
        "raw_dir": args.output_dir,
    }
    plan = build_download_plan(
        verification_config,
        dataset=args.dataset,
        symbol=args.symbol,
        start_date=args.date,
        end_date=args.date,
        max_files=1,
        template_name=args.template_name,
    )
    item = plan[0]

    if not args.execute:
        print("Bybit archive verification dry run")
        if args.template_name:
            print(f"template_name: {args.template_name}")
        print(f"url: {item.url}")
        print(f"target_path: {item.raw_path}")
        return

    timeout = float(config.get("timeout_seconds", 30))
    result = download_file(item, timeout=timeout)
    extract_dir = result.plan.raw_path.parent / "extracted"
    if args.dataset == "orderbook":
        inspection = inspect_jsonl_data_archive(
            result.plan.raw_path,
            extract_dir=extract_dir,
        )
        print("Bybit orderbook archive verification")
        print(f"local_zip_path: {inspection.archive_path}")
        print(f"detected_archive_type: {inspection.archive_type}")
        print(f"extracted_file_name: {inspection.read_path.name}")
        print(f"read_file_path: {inspection.read_path}")
        print(f"detected_keys: {json.dumps(inspection.detected_keys)}")
        print(f"first_snapshot_bid_levels: {inspection.first_bid_levels}")
        print(f"first_snapshot_ask_levels: {inspection.first_ask_levels}")
        print("first_5_json_objects:")
        print(json.dumps(inspection.first_objects, indent=2))
        return

    inspection = inspect_csv_archive(result.plan.raw_path, extract_dir=extract_dir)

    print("Bybit archive verification")
    print(f"local_file_path: {inspection.archive_path}")
    print(f"detected_archive_type: {inspection.archive_type}")
    print(f"read_file_path: {inspection.read_path}")
    print(f"row_count: {inspection.row_count}")
    print(f"columns: {inspection.columns}")
    print("first_5_rows:")
    print(json.dumps(inspection.first_rows, indent=2))


if __name__ == "__main__":
    main()
