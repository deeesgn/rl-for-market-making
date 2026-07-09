from __future__ import annotations

from datetime import date
from pathlib import Path

from scripts.check_orderbook_coverage import (
    compute_orderbook_coverage,
    expected_parquet_path,
    write_missing_dates,
)


def test_orderbook_coverage_detects_missing_dates(tmp_path: Path) -> None:
    input_dir = tmp_path / "processed" / "BTCUSDT"
    input_dir.mkdir(parents=True)
    expected_parquet_path(
        input_dir=input_dir,
        symbol="BTCUSDT",
        current_date=date_from_string("2025-01-01"),
    ).write_bytes(b"day 1")
    expected_parquet_path(
        input_dir=input_dir,
        symbol="BTCUSDT",
        current_date=date_from_string("2025-01-03"),
    ).write_bytes(b"day 3")

    coverage = compute_orderbook_coverage(
        input_dir=input_dir,
        symbol="BTCUSDT",
        start_date="2025-01-01",
        end_date="2025-01-03",
    )

    assert coverage.expected_dates == 3
    assert coverage.existing_parquet_files == 2
    assert [current_date.isoformat() for current_date in coverage.missing_dates] == [
        "2025-01-02"
    ]
    assert coverage.first_existing_date.isoformat() == "2025-01-01"
    assert coverage.last_existing_date.isoformat() == "2025-01-03"
    assert coverage.coverage_percentage == 66.6667
    assert coverage.total_size == len(b"day 1") + len(b"day 3")

    missing_output = tmp_path / "missing.txt"
    write_missing_dates(missing_output, coverage.missing_dates)

    assert missing_output.read_text(encoding="utf-8") == "2025-01-02\n"


def date_from_string(value: str) -> date:
    return date.fromisoformat(value)
