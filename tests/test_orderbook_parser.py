from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from rl_mm.data.orderbook_parser import (
    convert_orderbook_zip_to_parquet,
    iter_orderbook_messages_from_zip,
    iter_sampled_orderbook_rows,
)


def test_iter_orderbook_messages_from_zip_streams_data_lines(tmp_path: Path) -> None:
    zip_path = write_orderbook_zip(
        tmp_path / "orderbook.zip",
        [
            orderbook_message(
                "snapshot",
                1_735_689_600_100,
                bids=[["100", "1"]],
                asks=[["101", "2"]],
            ),
            orderbook_message("delta", 1_735_689_600_200, bids=[["100", "0.5"]], asks=[]),
        ],
    )

    messages = list(iter_orderbook_messages_from_zip(zip_path))

    assert len(messages) == 2
    assert messages[0]["type"] == "snapshot"
    assert messages[1]["data"]["b"] == [["100", "0.5"]]


def test_iter_sampled_orderbook_rows_applies_snapshot_delta_and_removal() -> None:
    messages = [
        orderbook_message(
            "snapshot",
            1_735_689_600_100,
            bids=[["100", "1"], ["99", "2"]],
            asks=[["101", "1"], ["102", "2"]],
            update_id=1,
            sequence=10,
        ),
        orderbook_message(
            "delta",
            1_735_689_600_900,
            bids=[["100", "0"], ["98", "3"]],
            asks=[["101", "4"]],
            update_id=2,
            sequence=11,
        ),
        orderbook_message(
            "delta",
            1_735_689_602_100,
            bids=[["103", "1"]],
            asks=[["104", "0.5"]],
            update_id=3,
            sequence=12,
        ),
    ]

    rows = list(iter_sampled_orderbook_rows(messages, symbol="BTCUSDT", depth=2, frequency="1s"))

    assert len(rows) == 3
    assert rows[0]["timestamp"] == "2025-01-01T00:00:00+00:00"
    assert rows[0]["bid_price_1"] == 99.0
    assert rows[0]["bid_size_1"] == 2.0
    assert rows[0]["bid_price_2"] == 98.0
    assert rows[0]["ask_price_1"] == 101.0
    assert rows[0]["ask_size_1"] == 4.0
    assert rows[0]["update_id"] == 2

    assert rows[1]["timestamp"] == "2025-01-01T00:00:01+00:00"
    assert rows[1]["bid_price_1"] == 99.0
    assert rows[1]["sequence"] == 11

    assert rows[2]["timestamp"] == "2025-01-01T00:00:02+00:00"
    assert rows[2]["bid_price_1"] == 103.0
    assert rows[2]["ask_price_1"] == 101.0
    assert rows[2]["mid_price"] == 102.0
    assert rows[2]["spread"] == -2.0


def test_convert_orderbook_zip_to_parquet_writes_expected_columns(tmp_path: Path) -> None:
    zip_path = write_orderbook_zip(
        tmp_path / "orderbook.zip",
        [
            orderbook_message(
                "snapshot",
                1_735_689_600_000,
                bids=[[str(100 - index), "1"] for index in range(12)],
                asks=[[str(101 + index), "2"] for index in range(12)],
            )
        ],
    )
    output_path = tmp_path / "processed.parquet"

    result = convert_orderbook_zip_to_parquet(
        zip_path,
        output_path=output_path,
        symbol="BTCUSDT",
        depth=10,
        frequency="1s",
    )

    dataframe = pd.read_parquet(output_path)
    assert result.row_count == 1
    assert dataframe.loc[0, "bid_price_1"] == 100.0
    assert dataframe.loc[0, "bid_price_10"] == 91.0
    assert dataframe.loc[0, "ask_price_1"] == 101.0
    assert dataframe.loc[0, "ask_price_10"] == 110.0
    assert dataframe.loc[0, "orderbook_imbalance"] == 10 / 30


def orderbook_message(
    message_type: str,
    timestamp_ms: int,
    *,
    bids: list[list[str]],
    asks: list[list[str]],
    update_id: int = 1,
    sequence: int = 1,
) -> dict:
    return {
        "topic": "orderbook.500.BTCUSDT",
        "type": message_type,
        "ts": timestamp_ms,
        "cts": timestamp_ms,
        "data": {
            "s": "BTCUSDT",
            "b": bids,
            "a": asks,
            "u": update_id,
            "seq": sequence,
        },
    }


def write_orderbook_zip(path: Path, messages: list[dict]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "2025-01-01_BTCUSDT_ob500.data",
            "\n".join(json.dumps(message) for message in messages),
        )
    return path
