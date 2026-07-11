from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from scripts.check_orderbook_ready import (
    REQUIRED_COLUMNS,
    check_orderbook_ready,
    expected_parquet_path,
)


def test_valid_orderbook_files_are_training_ready(tmp_path: Path) -> None:
    input_dir = tmp_path / "BTCUSDT"
    write_orderbook_parquet(input_dir, "2025-01-01", valid_orderbook_frame())

    report = check_orderbook_ready(
        input_dir=input_dir,
        symbol="BTCUSDT",
        start_date="2025-01-01",
        end_date="2025-01-01",
    )

    assert report["total_files"] == 1
    assert report["missing_dates"] == []
    assert report["schema_ok"] is True
    assert report["bad_files_count"] == 0
    assert report["total_rows"] == 2
    assert report["training_ready"] is True


def test_missing_orderbook_date_is_detected(tmp_path: Path) -> None:
    input_dir = tmp_path / "BTCUSDT"
    write_orderbook_parquet(input_dir, "2025-01-01", valid_orderbook_frame())

    report = check_orderbook_ready(
        input_dir=input_dir,
        symbol="BTCUSDT",
        start_date="2025-01-01",
        end_date="2025-01-02",
    )

    assert report["missing_dates"] == ["2025-01-02"]
    assert report["training_ready"] is False


def test_schema_mismatch_is_detected(tmp_path: Path) -> None:
    input_dir = tmp_path / "BTCUSDT"
    write_orderbook_parquet(input_dir, "2025-01-01", valid_orderbook_frame())
    mismatched = valid_orderbook_frame().drop(columns="sequence")
    write_orderbook_parquet(input_dir, "2025-01-02", mismatched)

    report = check_orderbook_ready(
        input_dir=input_dir,
        symbol="BTCUSDT",
        start_date="2025-01-01",
        end_date="2025-01-02",
    )

    assert report["schema_ok"] is False
    assert report["bad_files_count"] == 1
    assert report["training_ready"] is False


def test_invalid_best_bid_ask_is_detected(tmp_path: Path) -> None:
    input_dir = tmp_path / "BTCUSDT"
    invalid = valid_orderbook_frame()
    invalid.loc[0, "bid_price_1"] = invalid.loc[0, "ask_price_1"]
    write_orderbook_parquet(input_dir, "2025-01-01", invalid)

    report = check_orderbook_ready(
        input_dir=input_dir,
        symbol="BTCUSDT",
        start_date="2025-01-01",
        end_date="2025-01-01",
    )

    assert report["bad_files_count"] == 1
    assert "invalid_best_bid_ask" in report["bad_files"][0]["issues"]
    assert report["training_ready"] is False


def valid_orderbook_frame() -> pd.DataFrame:
    rows = {
        "timestamp": [
            "2025-01-01T00:00:00+00:00",
            "2025-01-01T00:00:01+00:00",
        ],
        "symbol": ["BTCUSDT", "BTCUSDT"],
    }
    for level in range(1, 11):
        rows[f"bid_price_{level}"] = [100.0 - level, 100.5 - level]
        rows[f"bid_size_{level}"] = [1.0, 1.5]
        rows[f"ask_price_{level}"] = [100.0 + level, 100.5 + level]
        rows[f"ask_size_{level}"] = [1.0, 1.5]
    rows.update(
        {
            "mid_price": [100.0, 100.5],
            "spread": [2.0, 2.0],
            "orderbook_imbalance": [0.5, 0.5],
            "update_id": [1, 2],
            "sequence": [10, 11],
        }
    )
    frame = pd.DataFrame(rows)
    return frame[REQUIRED_COLUMNS]


def write_orderbook_parquet(input_dir: Path, day: str, frame: pd.DataFrame) -> Path:
    path = expected_parquet_path(
        input_dir=input_dir,
        symbol="BTCUSDT",
        current_date=date.fromisoformat(day),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path
