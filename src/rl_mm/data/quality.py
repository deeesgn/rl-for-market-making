"""Quality checks for processed Bybit parquet files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rl_mm.data.schema import get_schema, parse_float, parse_timestamp, validate_records


@dataclass(frozen=True)
class ProcessedDataQualityReport:
    row_count: int
    columns: tuple[str, ...]
    timestamp_min: str
    timestamp_max: str
    missing_values: dict[str, int]
    duplicate_timestamp_count: int
    numeric_summary: dict[str, dict[str, float]]


def load_processed_parquet(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed data file not found: {path}. Run `make convert-bybit-sample` first."
        )
    return pd.read_parquet(path)


def check_processed_data(path: Path, *, dataset: str) -> ProcessedDataQualityReport:
    dataframe = load_processed_parquet(path)
    return summarize_processed_dataframe(dataframe, dataset=dataset)


def summarize_processed_dataframe(
    dataframe: pd.DataFrame,
    *,
    dataset: str,
) -> ProcessedDataQualityReport:
    schema = get_schema(dataset)
    records = dataframe.to_dict("records")
    validate_records(records, schema)

    timestamps = [parse_timestamp(row[schema.timestamp_column]) for row in records]
    duplicate_timestamp_count = int(dataframe.duplicated(subset=[schema.timestamp_column]).sum())
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

    return ProcessedDataQualityReport(
        row_count=len(records),
        columns=tuple(str(column) for column in dataframe.columns),
        timestamp_min=min(timestamps).isoformat(),
        timestamp_max=max(timestamps).isoformat(),
        missing_values={
            str(column): int(count) for column, count in dataframe.isna().sum().items()
        },
        duplicate_timestamp_count=duplicate_timestamp_count,
        numeric_summary=numeric_summary,
    )


def print_quality_report(report: ProcessedDataQualityReport) -> None:
    print(f"row_count: {report.row_count}")
    print(f"columns: {', '.join(report.columns)}")
    print(f"timestamp_range: {report.timestamp_min} -> {report.timestamp_max}")
    print("missing_values:")
    for column, count in report.missing_values.items():
        print(f"  {column}: {count}")
    print(f"duplicate_timestamp_count: {report.duplicate_timestamp_count}")
    print("price_size_summary:")
    for column, values in report.numeric_summary.items():
        print(
            f"  {column}: min={values['min']:.8f} "
            f"max={values['max']:.8f} mean={values['mean']:.8f}"
        )
