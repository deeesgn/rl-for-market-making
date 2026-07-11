"""Validate processed Bybit orderbook parquet files for training readiness."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REQUIRED_COLUMNS = (
    ["timestamp", "symbol"]
    + [f"bid_price_{level}" for level in range(1, 11)]
    + [f"bid_size_{level}" for level in range(1, 11)]
    + [f"ask_price_{level}" for level in range(1, 11)]
    + [f"ask_size_{level}" for level in range(1, 11)]
    + [
        "mid_price",
        "spread",
        "orderbook_imbalance",
        "update_id",
        "sequence",
    ]
)
VALIDATION_COLUMNS = [
    "timestamp",
    "bid_price_1",
    "ask_price_1",
    "mid_price",
    "spread",
    "orderbook_imbalance",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check orderbook parquet training readiness.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    args = parser.parse_args()

    report = check_orderbook_ready(
        input_dir=args.input_dir,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print_summary(report)
    report_path = args.input_dir.parent / f"orderbook_ready_{args.start_date[:4]}.json"
    write_report(report_path, report)
    print(f"report_path: {report_path}")


def check_orderbook_ready(
    *,
    input_dir: Path,
    symbol: str,
    start_date: str,
    end_date: str,
) -> dict:
    expected_dates = list(iter_dates(start_date, end_date))
    parquet_files = sorted(input_dir.glob("*.parquet")) if input_dir.is_dir() else []
    missing_dates = [
        current_date.isoformat()
        for current_date in expected_dates
        if not expected_parquet_path(
            input_dir=input_dir,
            symbol=symbol,
            current_date=current_date,
        ).exists()
    ]

    reference_columns: list[str] | None = None
    schema_ok = bool(parquet_files)
    bad_files = []
    total_rows = 0

    for path in parquet_files:
        row_count, columns, issues = validate_file(path, reference_columns=reference_columns)
        if reference_columns is None:
            reference_columns = columns
        if columns != reference_columns or any(
            issue.startswith(("schema_mismatch", "missing_required_columns"))
            for issue in issues
        ):
            schema_ok = False
        total_rows += row_count
        status = "ok" if not issues else "bad"
        print(f"{path.name}: row_count={row_count} status={status}")
        if issues:
            bad_files.append({"file": str(path), "issues": issues})

    training_ready = (
        len(parquet_files) == len(expected_dates)
        and not missing_dates
        and schema_ok
        and not bad_files
    )
    return {
        "symbol": symbol.upper(),
        "start_date": start_date,
        "end_date": end_date,
        "expected_files": len(expected_dates),
        "total_files": len(parquet_files),
        "missing_dates": missing_dates,
        "schema_ok": schema_ok,
        "bad_files_count": len(bad_files),
        "bad_files": bad_files,
        "total_rows": total_rows,
        "training_ready": training_ready,
    }


def validate_file(
    path: Path,
    *,
    reference_columns: list[str] | None,
) -> tuple[int, list[str], list[str]]:
    try:
        parquet_file = pq.ParquetFile(path)
        columns = parquet_file.schema_arrow.names
        row_count = parquet_file.metadata.num_rows
    except Exception as error:  # noqa: BLE001 - report unreadable files without stopping.
        return 0, [], [f"unreadable_parquet: {error}"]

    issues = []
    if reference_columns is not None and columns != reference_columns:
        issues.append("schema_mismatch")
    missing_required = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing_required:
        issues.append("missing_required_columns: " + ", ".join(missing_required))
        return row_count, columns, issues
    if row_count == 0:
        issues.append("empty_file")
        return row_count, columns, issues

    try:
        frame = pd.read_parquet(path, columns=VALIDATION_COLUMNS)
    except Exception as error:  # noqa: BLE001 - keep validating subsequent daily files.
        issues.append(f"unreadable_data: {error}")
        return row_count, columns, issues
    timestamps = frame["timestamp"]
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        issues.append("timestamps_not_sorted")
    if timestamps.duplicated().any():
        issues.append("duplicate_timestamps")
    if not (frame["bid_price_1"] < frame["ask_price_1"]).all():
        issues.append("invalid_best_bid_ask")
    if not (frame["spread"] > 0).all():
        issues.append("non_positive_spread")
    if frame["mid_price"].isna().any():
        issues.append("null_mid_price")
    if not frame["orderbook_imbalance"].between(0, 1, inclusive="both").all():
        issues.append("invalid_orderbook_imbalance")
    return row_count, columns, issues


def print_summary(report: dict) -> None:
    print("Orderbook readiness summary")
    print(f"total_files: {report['total_files']}")
    print(
        "missing_dates: "
        + (", ".join(report["missing_dates"]) if report["missing_dates"] else "none")
    )
    print(f"schema_ok: {'yes' if report['schema_ok'] else 'no'}")
    print(f"bad_files_count: {report['bad_files_count']}")
    print(f"total_rows: {report['total_rows']}")
    print(f"training_ready: {str(report['training_ready']).lower()}")


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expected_parquet_path(*, input_dir: Path, symbol: str, current_date: date) -> Path:
    return input_dir / (
        f"{symbol.upper()}_{current_date.isoformat()}_orderbook_top10_1s.parquet"
    )


def iter_dates(start_date: str, end_date: str):
    current = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    if end < current:
        raise ValueError("end_date must be on or after start_date")
    while current <= end:
        yield current
        current += timedelta(days=1)


if __name__ == "__main__":
    main()
