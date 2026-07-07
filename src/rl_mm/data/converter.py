"""Convert local Bybit CSV files into normalized processed parquet files."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Any

import pandas as pd

from rl_mm.data.schema import (
    DatasetSchema,
    get_schema,
    parse_float,
    parse_timestamp,
    validate_records,
)

TRADE_EXTRA_COLUMNS = (
    "tickDirection",
    "trdMatchID",
    "grossValue",
    "homeNotional",
    "foreignNotional",
)
TRADE_EXTRA_NUMERIC_COLUMNS = {"grossValue", "homeNotional", "foreignNotional"}


def load_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    with open_csv_text(path) as file:
        return list(csv.DictReader(file))


def open_csv_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def normalize_records(records: list[dict[str, Any]], *, dataset: str) -> pd.DataFrame:
    schema = get_schema(dataset)
    validate_records(records, schema)
    normalized = [normalize_record(record, schema=schema) for record in records]
    return pd.DataFrame(normalized, columns=normalized_columns(schema, normalized))


def normalize_record(record: dict[str, Any], *, schema: DatasetSchema) -> dict[str, Any]:
    normalized = {column: record[column] for column in schema.required_columns}
    timestamp = parse_timestamp(normalized[schema.timestamp_column])
    normalized[schema.timestamp_column] = pd.Timestamp(timestamp).tz_convert("UTC").isoformat()
    normalized["symbol"] = str(normalized["symbol"]).upper()

    if schema.name == "trades":
        normalized["side"] = str(normalized["side"]).lower()
        for column in TRADE_EXTRA_COLUMNS:
            value = record.get(column)
            if value is None or value == "":
                continue
            if column in TRADE_EXTRA_NUMERIC_COLUMNS:
                normalized[column] = parse_float(
                    value,
                    schema=schema.name,
                    row_index=1,
                    column=column,
                )
            else:
                normalized[column] = value

    for column in schema.numeric_columns:
        normalized[column] = parse_float(
            normalized[column],
            schema=schema.name,
            row_index=1,
            column=column,
        )

    return normalized


def normalized_columns(
    schema: DatasetSchema,
    records: list[dict[str, Any]],
) -> list[str]:
    columns = list(schema.required_columns)
    if schema.name == "trades":
        for column in TRADE_EXTRA_COLUMNS:
            if any(column in record for record in records):
                columns.append(column)
    return columns


def convert_csv_to_parquet(input_path: Path, *, dataset: str, output_path: Path) -> Path:
    records = load_csv_records(input_path)
    dataframe = normalize_records(records, dataset=dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False)
    return output_path
