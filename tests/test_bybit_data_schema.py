from pathlib import Path

import pytest

from rl_mm.data.bybit_downloader import build_download_plan, validate_local_csv
from rl_mm.data.schema import (
    ORDERBOOK_SCHEMA,
    TRADES_SCHEMA,
    SchemaValidationError,
    validate_records,
)
from scripts.inspect_bybit_data import summarize_records


def test_trade_schema_accepts_expected_columns() -> None:
    validate_records(
        [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "symbol": "BTCUSDT",
                "side": "buy",
                "price": "42000.5",
                "size": "0.1",
            }
        ],
        TRADES_SCHEMA,
    )


def test_orderbook_schema_rejects_missing_required_column() -> None:
    with pytest.raises(SchemaValidationError, match="missing columns"):
        validate_records(
            [
                {
                    "timestamp": "2024-01-01T00:00:00Z",
                    "symbol": "BTCUSDT",
                    "bid_price": "42000.0",
                    "bid_size": "1.0",
                    "ask_price": "42001.0",
                }
            ],
            ORDERBOOK_SCHEMA,
        )


def test_build_download_plan_uses_storage_layout() -> None:
    plan = build_download_plan(
        {
            "symbol": "BTCUSDT",
            "start_date": "2024-01-01",
            "end_date": "2024-01-02",
            "datasets": ["trades"],
            "raw_dir": "data/raw/bybit",
            "processed_dir": "data/processed/bybit",
            "url_templates": {"trades": "https://example.test/{symbol}/{filename}"},
        }
    )

    assert len(plan) == 2
    assert plan[0].raw_path == Path("data/raw/bybit/trades/BTCUSDT_trades_2024-01-01.csv.gz")
    assert plan[0].processed_path == Path("data/processed/bybit/trades/BTCUSDT_2024-01-01.parquet")
    assert plan[0].url == "https://example.test/BTCUSDT/BTCUSDT_trades_2024-01-01.csv.gz"


def test_validate_local_csv_and_inspect_summary(tmp_path: Path) -> None:
    csv_path = tmp_path / "trades.csv"
    csv_path.write_text(
        "\n".join(
            [
                "timestamp,symbol,side,price,size",
                "2024-01-01T00:00:00Z,BTCUSDT,buy,42000.0,0.1",
                "2024-01-01T00:00:01Z,BTCUSDT,sell,42002.0,0.3",
            ]
        ),
        encoding="utf-8",
    )

    validate_local_csv(csv_path, dataset="trades")
    records = [
        {
            "timestamp": "2024-01-01T00:00:00Z",
            "symbol": "BTCUSDT",
            "side": "buy",
            "price": "42000.0",
            "size": "0.1",
        },
        {
            "timestamp": "2024-01-01T00:00:01Z",
            "symbol": "BTCUSDT",
            "side": "sell",
            "price": "42002.0",
            "size": "0.3",
        },
    ]

    summary = summarize_records(records, dataset="trades")

    assert summary["row_count"] == 2
    assert summary["missing_values"]["price"] == 0
    assert summary["numeric_summary"]["price"]["mean"] == 42001.0
