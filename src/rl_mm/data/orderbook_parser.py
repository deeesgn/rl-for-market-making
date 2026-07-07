"""Stream Bybit quote-saver orderbook archives into sampled Top-N parquet."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd


@dataclass(frozen=True)
class OrderBookConversionResult:
    output_path: Path
    row_count: int
    timestamp_min: str | None
    timestamp_max: str | None


def convert_orderbook_zip_to_parquet(
    input_path: Path,
    *,
    output_path: Path,
    symbol: str,
    depth: int = 10,
    frequency: str = "1s",
) -> OrderBookConversionResult:
    dataframe = orderbook_zip_to_dataframe(
        input_path,
        symbol=symbol,
        depth=depth,
        frequency=frequency,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)
    timestamps = dataframe["timestamp"].tolist() if "timestamp" in dataframe else []
    return OrderBookConversionResult(
        output_path=output_path,
        row_count=int(len(dataframe)),
        timestamp_min=min(timestamps) if timestamps else None,
        timestamp_max=max(timestamps) if timestamps else None,
    )


def orderbook_zip_to_dataframe(
    input_path: Path,
    *,
    symbol: str,
    depth: int = 10,
    frequency: str = "1s",
) -> pd.DataFrame:
    rows = list(
        iter_sampled_orderbook_rows(
            iter_orderbook_messages_from_zip(input_path),
            symbol=symbol,
            depth=depth,
            frequency=frequency,
        )
    )
    return pd.DataFrame(rows, columns=orderbook_columns(depth))


def iter_orderbook_messages_from_zip(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Orderbook ZIP file not found: {path}")

    with zipfile.ZipFile(path) as archive:
        members = [
            member
            for member in archive.namelist()
            if not member.endswith("/") and member.lower().endswith(".data")
        ]
        if not members:
            raise ValueError(f"No .data file found inside {path}")

        with archive.open(members[0]) as binary_file:
            text_file = io.TextIOWrapper(binary_file, encoding="utf-8")
            for line_number, line in enumerate(text_file, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    message = json.loads(text)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Could not parse JSON line {line_number} in {path}: {error}"
                    ) from error
                if not isinstance(message, dict):
                    raise ValueError(f"Expected JSON object on line {line_number} in {path}")
                yield message


def iter_sampled_orderbook_rows(
    messages: Iterable[dict[str, Any]],
    *,
    symbol: str,
    depth: int = 10,
    frequency: str = "1s",
) -> Iterator[dict[str, Any]]:
    frequency_seconds = parse_frequency_seconds(frequency)
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    pending_second: datetime | None = None
    pending_row: dict[str, Any] | None = None

    for message in messages:
        data = message.get("data")
        if not isinstance(data, dict):
            continue

        message_symbol = str(data.get("s") or symbol).upper()
        if message_symbol != symbol.upper():
            continue

        apply_orderbook_message(message, bids=bids, asks=asks)
        if not bids or not asks:
            continue

        timestamp = parse_orderbook_timestamp(message.get("cts") or message.get("ts"))
        sample_second = floor_timestamp(timestamp, frequency_seconds)
        row = build_orderbook_row(
            bids=bids,
            asks=asks,
            timestamp=sample_second,
            symbol=message_symbol,
            depth=depth,
            update_id=data.get("u"),
            sequence=data.get("seq"),
        )

        if pending_second is None:
            pending_second = sample_second
            pending_row = row
            continue

        if sample_second == pending_second:
            pending_row = row
            continue

        if pending_row is not None:
            yield pending_row
            yield from fill_missing_seconds(
                last_row=pending_row,
                start_after=pending_second,
                stop_before=sample_second,
                frequency_seconds=frequency_seconds,
            )
        pending_second = sample_second
        pending_row = row

    if pending_row is not None:
        yield pending_row


def apply_orderbook_message(
    message: dict[str, Any],
    *,
    bids: dict[float, float],
    asks: dict[float, float],
) -> None:
    data = message.get("data")
    if not isinstance(data, dict):
        return

    if message.get("type") == "snapshot":
        bids.clear()
        asks.clear()
        bids.update(levels_to_dict(data.get("b", [])))
        asks.update(levels_to_dict(data.get("a", [])))
        return

    update_side(bids, data.get("b", []))
    update_side(asks, data.get("a", []))


def levels_to_dict(levels: Any) -> dict[float, float]:
    side: dict[float, float] = {}
    if not isinstance(levels, list):
        return side
    for level in levels:
        price, size = parse_level(level)
        if size > 0:
            side[price] = size
    return side


def update_side(side: dict[float, float], levels: Any) -> None:
    if not isinstance(levels, list):
        return
    for level in levels:
        price, size = parse_level(level)
        if size == 0:
            side.pop(price, None)
        else:
            side[price] = size


def parse_level(level: Any) -> tuple[float, float]:
    if not isinstance(level, list | tuple) or len(level) < 2:
        raise ValueError(f"Invalid orderbook level: {level}")
    return float(level[0]), float(level[1])


def build_orderbook_row(
    *,
    bids: dict[float, float],
    asks: dict[float, float],
    timestamp: datetime,
    symbol: str,
    depth: int,
    update_id: Any,
    sequence: Any,
) -> dict[str, Any]:
    bid_levels = sorted(bids.items(), key=lambda item: item[0], reverse=True)[:depth]
    ask_levels = sorted(asks.items(), key=lambda item: item[0])[:depth]

    row: dict[str, Any] = {
        "timestamp": timestamp.isoformat(),
        "symbol": symbol.upper(),
    }
    add_side_columns(row, prefix="bid", levels=bid_levels, depth=depth)
    add_side_columns(row, prefix="ask", levels=ask_levels, depth=depth)

    best_bid = bid_levels[0][0] if bid_levels else None
    best_ask = ask_levels[0][0] if ask_levels else None
    row["mid_price"] = (
        (best_bid + best_ask) / 2
        if best_bid is not None and best_ask is not None
        else None
    )
    row["spread"] = best_ask - best_bid if best_bid is not None and best_ask is not None else None
    row["orderbook_imbalance"] = orderbook_imbalance(bid_levels, ask_levels)
    row["update_id"] = update_id
    row["sequence"] = sequence
    return row


def add_side_columns(
    row: dict[str, Any],
    *,
    prefix: str,
    levels: list[tuple[float, float]],
    depth: int,
) -> None:
    for index in range(depth):
        price, size = levels[index] if index < len(levels) else (None, None)
        row[f"{prefix}_price_{index + 1}"] = price
        row[f"{prefix}_size_{index + 1}"] = size


def orderbook_imbalance(
    bid_levels: list[tuple[float, float]],
    ask_levels: list[tuple[float, float]],
) -> float | None:
    bid_size = sum(size for _, size in bid_levels)
    ask_size = sum(size for _, size in ask_levels)
    denominator = bid_size + ask_size
    if denominator == 0:
        return None
    return bid_size / denominator


def fill_missing_seconds(
    *,
    last_row: dict[str, Any],
    start_after: datetime,
    stop_before: datetime,
    frequency_seconds: int,
) -> Iterator[dict[str, Any]]:
    current = datetime.fromtimestamp(
        int(start_after.timestamp()) + frequency_seconds,
        tz=timezone.utc,
    )
    while current < stop_before:
        row = dict(last_row)
        row["timestamp"] = current.isoformat()
        yield row
        current = datetime.fromtimestamp(
            int(current.timestamp()) + frequency_seconds,
            tz=timezone.utc,
        )


def parse_orderbook_timestamp(value: Any) -> datetime:
    if value is None:
        raise ValueError("Orderbook message is missing ts/cts timestamp")
    return datetime.fromtimestamp(float(value) / 1_000, tz=timezone.utc)


def floor_timestamp(timestamp: datetime, frequency_seconds: int) -> datetime:
    epoch_seconds = int(timestamp.timestamp())
    floored_seconds = epoch_seconds - (epoch_seconds % frequency_seconds)
    return datetime.fromtimestamp(floored_seconds, tz=timezone.utc)


def parse_frequency_seconds(frequency: str) -> int:
    seconds = int(pd.Timedelta(frequency).total_seconds())
    if seconds <= 0:
        raise ValueError("frequency must be positive")
    return seconds


def orderbook_columns(depth: int) -> list[str]:
    columns = ["timestamp", "symbol"]
    columns.extend(f"bid_price_{index}" for index in range(1, depth + 1))
    columns.extend(f"bid_size_{index}" for index in range(1, depth + 1))
    columns.extend(f"ask_price_{index}" for index in range(1, depth + 1))
    columns.extend(f"ask_size_{index}" for index in range(1, depth + 1))
    columns.extend(["mid_price", "spread", "orderbook_imbalance", "update_id", "sequence"])
    return columns
