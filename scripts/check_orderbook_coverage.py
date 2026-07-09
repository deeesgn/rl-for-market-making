"""Check daily processed Bybit orderbook parquet coverage."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path


@dataclass(frozen=True)
class OrderBookCoverage:
    expected_dates: int
    existing_parquet_files: int
    missing_dates: list[date]
    first_existing_date: date | None
    last_existing_date: date | None
    coverage_percentage: float
    total_size: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Check processed orderbook coverage.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--missing-output", type=Path)
    args = parser.parse_args()

    coverage = compute_orderbook_coverage(
        input_dir=args.input_dir,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print_orderbook_coverage(coverage)
    if args.missing_output is not None:
        write_missing_dates(args.missing_output, coverage.missing_dates)
        print(f"missing_output: {args.missing_output}")


def compute_orderbook_coverage(
    *,
    input_dir: Path,
    symbol: str,
    start_date: str,
    end_date: str,
) -> OrderBookCoverage:
    expected = list(iter_dates(start_date, end_date))
    existing_dates = []
    total_size = 0
    for current_date in expected:
        path = expected_parquet_path(input_dir=input_dir, symbol=symbol, current_date=current_date)
        if path.exists():
            existing_dates.append(current_date)
            total_size += path.stat().st_size

    missing_dates = [
        current_date for current_date in expected if current_date not in existing_dates
    ]
    existing_count = len(existing_dates)
    expected_count = len(expected)
    coverage_percentage = (existing_count / expected_count * 100.0) if expected_count else 100.0
    return OrderBookCoverage(
        expected_dates=expected_count,
        existing_parquet_files=existing_count,
        missing_dates=missing_dates,
        first_existing_date=min(existing_dates) if existing_dates else None,
        last_existing_date=max(existing_dates) if existing_dates else None,
        coverage_percentage=round(coverage_percentage, 4),
        total_size=total_size,
    )


def expected_parquet_path(*, input_dir: Path, symbol: str, current_date: date) -> Path:
    filename = f"{symbol.upper()}_{current_date.isoformat()}_orderbook_top10_1s.parquet"
    return input_dir / filename


def print_orderbook_coverage(coverage: OrderBookCoverage) -> None:
    print(f"expected_dates: {coverage.expected_dates}")
    print(f"existing_parquet_files: {coverage.existing_parquet_files}")
    print(
        "missing_dates: "
        + (
            ", ".join(current_date.isoformat() for current_date in coverage.missing_dates)
            if coverage.missing_dates
            else "none"
        )
    )
    print(
        "first_existing_date: "
        + (coverage.first_existing_date.isoformat() if coverage.first_existing_date else "none")
    )
    print(
        "last_existing_date: "
        + (coverage.last_existing_date.isoformat() if coverage.last_existing_date else "none")
    )
    print(f"coverage_percentage: {coverage.coverage_percentage}")
    print(f"total_size: {coverage.total_size}")


def write_missing_dates(path: Path, missing_dates: list[date]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(current_date.isoformat() for current_date in missing_dates)
        + ("\n" if missing_dates else ""),
        encoding="utf-8",
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
