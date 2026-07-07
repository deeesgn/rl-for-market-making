"""Inspect a local Bybit CSV or parquet file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from rl_mm.data.schema import get_schema, parse_float, parse_timestamp, validate_records


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as error:
            raise RuntimeError("Parquet inspection requires pandas to be installed.") from error
        return pd.read_parquet(path).to_dict("records")
    raise ValueError(f"Unsupported file type '{suffix}'. Expected .csv or .parquet.")


def summarize_records(records: list[dict[str, Any]], *, dataset: str) -> dict[str, Any]:
    schema = get_schema(dataset)
    validate_records(records, schema)
    columns = list(records[0]) if records else []
    timestamps = [parse_timestamp(row[schema.timestamp_column]) for row in records]
    missing_values = {
        column: sum(row.get(column) in (None, "") for row in records)
        for column in columns
    }
    numeric_summary = {}
    for column in (*schema.price_columns, *schema.size_columns):
        values = [
            parse_float(row[column], schema=schema.name, row_index=index, column=column)
            for index, row in enumerate(records, start=1)
        ]
        numeric_summary[column] = {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    return {
        "row_count": len(records),
        "columns": columns,
        "timestamp_min": min(timestamps),
        "timestamp_max": max(timestamps),
        "missing_values": missing_values,
        "numeric_summary": numeric_summary,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"row_count: {summary['row_count']}")
    print(f"columns: {', '.join(summary['columns'])}")
    print(f"timestamp_range: {summary['timestamp_min']} -> {summary['timestamp_max']}")
    print("missing_values:")
    for column, count in summary["missing_values"].items():
        print(f"  {column}: {count}")
    print("price_size_summary:")
    for column, values in summary["numeric_summary"].items():
        print(
            f"  {column}: min={values['min']:.8f} "
            f"max={values['max']:.8f} mean={values['mean']:.8f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local Bybit data.")
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--dataset", choices=["trades", "orderbook"], default="trades")
    args = parser.parse_args()

    records = load_records(args.path)
    summary = summarize_records(records, dataset=args.dataset)
    print_summary(summary)


if __name__ == "__main__":
    main()
