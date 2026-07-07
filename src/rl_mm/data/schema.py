"""Lightweight schemas for Bybit market data files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping


class SchemaValidationError(ValueError):
    """Raised when market data does not match an expected schema."""


@dataclass(frozen=True)
class DatasetSchema:
    """Simple tabular schema for local CSV/parquet validation."""

    name: str
    required_columns: tuple[str, ...]
    timestamp_column: str
    numeric_columns: tuple[str, ...]
    price_columns: tuple[str, ...]
    size_columns: tuple[str, ...]


TRADES_SCHEMA = DatasetSchema(
    name="trades",
    required_columns=("timestamp", "symbol", "side", "price", "size"),
    timestamp_column="timestamp",
    numeric_columns=("price", "size"),
    price_columns=("price",),
    size_columns=("size",),
)

ORDERBOOK_SCHEMA = DatasetSchema(
    name="orderbook",
    required_columns=(
        "timestamp",
        "symbol",
        "bid_price",
        "bid_size",
        "ask_price",
        "ask_size",
    ),
    timestamp_column="timestamp",
    numeric_columns=("bid_price", "bid_size", "ask_price", "ask_size"),
    price_columns=("bid_price", "ask_price"),
    size_columns=("bid_size", "ask_size"),
)

SCHEMAS = {
    TRADES_SCHEMA.name: TRADES_SCHEMA,
    ORDERBOOK_SCHEMA.name: ORDERBOOK_SCHEMA,
    "orderbook_snapshot": ORDERBOOK_SCHEMA,
    "orderbook_updates": ORDERBOOK_SCHEMA,
}


def get_schema(dataset: str) -> DatasetSchema:
    try:
        return SCHEMAS[dataset]
    except KeyError as error:
        known = ", ".join(sorted(SCHEMAS))
        message = f"Unknown dataset '{dataset}'. Expected one of: {known}"
        raise SchemaValidationError(message) from error


def validate_records(records: Iterable[Mapping[str, Any]], schema: DatasetSchema) -> None:
    """Validate required columns, timestamps, and numeric fields."""

    rows = list(records)
    if not rows:
        raise SchemaValidationError(f"{schema.name} data is empty.")

    missing_columns = set(schema.required_columns) - set(rows[0])
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise SchemaValidationError(f"{schema.name} data is missing columns: {missing}")

    for row_index, row in enumerate(rows, start=1):
        for column in schema.required_columns:
            value = row.get(column)
            if value is None or value == "":
                raise SchemaValidationError(
                    f"{schema.name} row {row_index} has missing value in '{column}'."
                )
        parse_timestamp(row[schema.timestamp_column])
        for column in schema.numeric_columns:
            parse_float(row[column], schema=schema.name, row_index=row_index, column=column)


def parse_timestamp(value: Any) -> datetime:
    """Parse ISO timestamps or millisecond/microsecond/nanosecond epoch values."""

    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        raise SchemaValidationError("Timestamp value is empty.")
    if text.isdigit():
        integer_value = int(text)
        if integer_value > 10**17:
            seconds = integer_value / 1_000_000_000
        elif integer_value > 10**14:
            seconds = integer_value / 1_000_000
        elif integer_value > 10**11:
            seconds = integer_value / 1_000
        else:
            seconds = integer_value
        return datetime.fromtimestamp(seconds)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise SchemaValidationError(f"Could not parse timestamp '{value}'.") from error


def parse_float(value: Any, *, schema: str, row_index: int, column: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise SchemaValidationError(
            f"{schema} row {row_index} has non-numeric value in '{column}': {value}"
        ) from error
