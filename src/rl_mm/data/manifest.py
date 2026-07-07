"""Dataset manifest builder for processed parquet files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from rl_mm.data.schema import parse_timestamp


@dataclass(frozen=True)
class ManifestEntry:
    file_path: str
    dataset_type: str | None
    row_count: int
    columns: tuple[str, ...]
    timestamp_min: str | None
    timestamp_max: str | None
    file_size_bytes: int


def build_dataset_manifest(input_dir: Path) -> dict[str, Any]:
    files = sorted(input_dir.rglob("*.parquet")) if input_dir.is_dir() else []
    entries = [inspect_parquet_file(path) for path in files]
    return {"input_dir": str(input_dir), "files": [asdict(entry) for entry in entries]}


def inspect_parquet_file(path: Path) -> ManifestEntry:
    dataframe = pd.read_parquet(path)
    timestamp_min = None
    timestamp_max = None
    if "timestamp" in dataframe.columns and not dataframe.empty:
        timestamps = [parse_timestamp(value) for value in dataframe["timestamp"].tolist()]
        timestamp_min = min(timestamps).isoformat()
        timestamp_max = max(timestamps).isoformat()

    return ManifestEntry(
        file_path=str(path),
        dataset_type=infer_dataset_type(path),
        row_count=int(len(dataframe)),
        columns=tuple(str(column) for column in dataframe.columns),
        timestamp_min=timestamp_min,
        timestamp_max=timestamp_max,
        file_size_bytes=path.stat().st_size,
    )


def infer_dataset_type(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    if "trades" in parts:
        return "trades"
    if "orderbook" in parts:
        return "orderbook"
    return None


def write_dataset_manifest(input_dir: Path, output_path: Path) -> Path:
    manifest = build_dataset_manifest(input_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
    return output_path
