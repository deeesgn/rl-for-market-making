from pathlib import Path

import pandas as pd
import pytest

from rl_mm.data.converter import convert_csv_to_parquet, load_csv_records, normalize_records


def test_normalize_trade_records() -> None:
    dataframe = normalize_records(
        [
            {
                "timestamp": "2024-01-01T00:00:00Z",
                "symbol": "btcusdt",
                "side": "BUY",
                "price": "42000.5",
                "size": "0.10",
            }
        ],
        dataset="trades",
    )

    assert dataframe.loc[0, "timestamp"] == "2024-01-01T00:00:00+00:00"
    assert dataframe.loc[0, "symbol"] == "BTCUSDT"
    assert dataframe.loc[0, "side"] == "buy"
    assert dataframe.loc[0, "price"] == 42000.5
    assert dataframe.loc[0, "size"] == 0.10


def test_convert_csv_to_parquet(tmp_path: Path) -> None:
    input_path = tmp_path / "trades.csv"
    output_path = tmp_path / "processed" / "trades.parquet"
    input_path.write_text(
        "\n".join(
            [
                "timestamp,symbol,side,price,size",
                "2024-01-01T00:00:00Z,btcusdt,BUY,42000.5,0.10",
                "2024-01-01T00:00:01Z,BTCUSDT,SELL,42001.0,0.05",
            ]
        ),
        encoding="utf-8",
    )

    result = convert_csv_to_parquet(input_path, dataset="trades", output_path=output_path)

    assert result == output_path
    assert output_path.is_file()
    dataframe = pd.read_parquet(output_path)
    assert list(dataframe.columns) == ["timestamp", "symbol", "side", "price", "size"]
    assert dataframe["symbol"].tolist() == ["BTCUSDT", "BTCUSDT"]
    assert dataframe["side"].tolist() == ["buy", "sell"]
    assert dataframe["price"].tolist() == [42000.5, 42001.0]


def test_convert_orderbook_csv_to_parquet(tmp_path: Path) -> None:
    input_path = tmp_path / "orderbook.csv"
    output_path = tmp_path / "orderbook.parquet"
    input_path.write_text(
        "\n".join(
            [
                "timestamp,symbol,bid_price,bid_size,ask_price,ask_size",
                "2024-01-01T00:00:00Z,btcusdt,42000.0,1.5,42001.0,1.1",
            ]
        ),
        encoding="utf-8",
    )

    convert_csv_to_parquet(input_path, dataset="orderbook", output_path=output_path)

    dataframe = pd.read_parquet(output_path)
    assert dataframe.loc[0, "symbol"] == "BTCUSDT"
    assert dataframe.loc[0, "bid_price"] == 42000.0
    assert dataframe.loc[0, "ask_size"] == 1.1


def test_missing_input_file_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        load_csv_records(tmp_path / "missing.csv")
