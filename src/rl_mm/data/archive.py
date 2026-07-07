"""Utilities for reading small market data archives."""

from __future__ import annotations

import csv
import gzip
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


@dataclass(frozen=True)
class ArchiveInspection:
    archive_path: Path
    archive_type: str
    read_path: Path
    row_count: int
    columns: list[str]
    first_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class JsonLinesArchiveInspection:
    archive_path: Path
    archive_type: str
    read_path: Path
    first_objects: list[dict[str, Any]]
    detected_keys: dict[str, list[str]]
    first_bid_levels: int | None
    first_ask_levels: int | None


class ArchiveError(ValueError):
    """Raised when an archive cannot be detected or read."""


def detect_archive_type(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".csv"):
        return "csv"
    if name.endswith(".csv.gz"):
        return "csv.gz"
    if name.endswith(".zip"):
        return "zip"
    raise ArchiveError(f"Unsupported archive type for {path}. Expected .csv, .csv.gz, or .zip.")


def inspect_csv_archive(
    path: Path,
    *,
    extract_dir: Path | None = None,
    preview_rows: int = 5,
) -> ArchiveInspection:
    if not path.is_file():
        raise FileNotFoundError(f"Archive file not found: {path}")

    archive_type = detect_archive_type(path)
    if archive_type == "zip":
        read_path = extract_first_csv_from_zip(path, extract_dir=extract_dir)
        rows, columns = read_csv_file(read_path, preview_rows=preview_rows)
    elif archive_type == "csv.gz":
        read_path = path
        rows, columns = read_gzip_csv_file(path, preview_rows=preview_rows)
    else:
        read_path = path
        rows, columns = read_csv_file(path, preview_rows=preview_rows)

    return ArchiveInspection(
        archive_path=path,
        archive_type=archive_type,
        read_path=read_path,
        row_count=rows["row_count"],
        columns=columns,
        first_rows=rows["first_rows"],
    )


def inspect_jsonl_data_archive(
    path: Path,
    *,
    extract_dir: Path | None = None,
    preview_rows: int = 5,
) -> JsonLinesArchiveInspection:
    if not path.is_file():
        raise FileNotFoundError(f"Archive file not found: {path}")

    archive_type = detect_archive_type(path)
    if archive_type != "zip":
        raise ArchiveError(f"Expected a .zip archive containing a .data file: {path}")

    read_path = extract_first_data_from_zip(path, extract_dir=extract_dir)
    first_objects = read_jsonl_preview(read_path, preview_rows=preview_rows)
    first_object = first_objects[0] if first_objects else {}
    data = first_object.get("data", {}) if isinstance(first_object.get("data"), dict) else {}

    return JsonLinesArchiveInspection(
        archive_path=path,
        archive_type=archive_type,
        read_path=read_path,
        first_objects=first_objects,
        detected_keys=detected_json_keys(first_object),
        first_bid_levels=count_levels(data.get("b")),
        first_ask_levels=count_levels(data.get("a")),
    )


def extract_first_csv_from_zip(path: Path, *, extract_dir: Path | None = None) -> Path:
    return extract_first_member_from_zip(path, suffix=".csv", extract_dir=extract_dir)


def extract_first_data_from_zip(path: Path, *, extract_dir: Path | None = None) -> Path:
    return extract_first_member_from_zip(path, suffix=".data", extract_dir=extract_dir)


def extract_first_member_from_zip(
    path: Path,
    *,
    suffix: str,
    extract_dir: Path | None = None,
) -> Path:
    output_dir = extract_dir or path.parent / f"{path.stem}_extracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(path) as archive:
        members = [
            member
            for member in archive.namelist()
            if not member.endswith("/") and member.lower().endswith(suffix)
        ]
        if not members:
            raise ArchiveError(f"No {suffix} file found inside {path}")

        member = members[0]
        output_path = output_dir / Path(member).name
        with archive.open(member) as source, output_path.open("wb") as target:
            target.write(source.read())
        return output_path


def read_csv_file(path: Path, *, preview_rows: int = 5) -> tuple[dict[str, Any], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return read_csv_stream(file, preview_rows=preview_rows)


def read_gzip_csv_file(path: Path, *, preview_rows: int = 5) -> tuple[dict[str, Any], list[str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as file:
        return read_csv_stream(file, preview_rows=preview_rows)


def read_csv_stream(file: TextIO, *, preview_rows: int = 5) -> tuple[dict[str, Any], list[str]]:
    reader = csv.DictReader(file)
    columns = list(reader.fieldnames or [])
    first_rows = []
    row_count = 0

    for row in reader:
        row_count += 1
        if len(first_rows) < preview_rows:
            first_rows.append(dict(row))

    return {"row_count": row_count, "first_rows": first_rows}, columns


def read_jsonl_preview(path: Path, *, preview_rows: int = 5) -> list[dict[str, Any]]:
    objects = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if len(objects) >= preview_rows:
                break
            text = line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as error:
                raise ArchiveError(
                    f"Could not parse JSON line {line_number} in {path}: {error}"
                ) from error
            if not isinstance(parsed, dict):
                raise ArchiveError(f"Expected JSON object on line {line_number} in {path}")
            objects.append(parsed)
    return objects


def detected_json_keys(first_object: dict[str, Any]) -> dict[str, list[str]]:
    data = first_object.get("data")
    return {
        "top_level": sorted(str(key) for key in first_object),
        "data": sorted(str(key) for key in data) if isinstance(data, dict) else [],
    }


def count_levels(value: Any) -> int | None:
    if isinstance(value, list):
        return len(value)
    return None
