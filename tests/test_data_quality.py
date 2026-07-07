from pathlib import Path

import pandas as pd
import pytest

from rl_mm.data.quality import check_processed_data


def test_check_processed_data_reports_quality_metrics(tmp_path: Path) -> None:
    parquet_path = tmp_path / "trades.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01T00:00:00+00:00",
                "symbol": "BTCUSDT",
                "side": "buy",
                "price": 42000.0,
                "size": 0.1,
            },
            {
                "timestamp": "2024-01-01T00:00:00+00:00",
                "symbol": "BTCUSDT",
                "side": "sell",
                "price": 42002.0,
                "size": 0.3,
            },
        ]
    ).to_parquet(parquet_path, index=False)

    report = check_processed_data(parquet_path, dataset="trades")

    assert report.row_count == 2
    assert report.columns == ("timestamp", "symbol", "side", "price", "size")
    assert report.duplicate_timestamp_count == 1
    assert report.missing_values["price"] == 0
    assert report.numeric_summary["price"]["mean"] == 42001.0
    assert report.numeric_summary["size"]["max"] == 0.3


def test_check_processed_data_missing_file_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make convert-bybit-sample"):
        check_processed_data(tmp_path / "missing.parquet", dataset="trades")
