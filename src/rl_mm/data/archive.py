"""Utilities for reading small CSV-based market data archives."""

from __future__ import annotations

import csv
import gzip
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


def extract_first_csv_from_zip(path: Path, *, extract_dir: Path | None = None) -> Path:
    output_dir = extract_dir or path.parent / f"{path.stem}_extracted"
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(path) as archive:
        csv_members = [
            member
            for member in archive.namelist()
            if not member.endswith("/") and member.lower().endswith(".csv")
        ]
        if not csv_members:
            raise ArchiveError(f"No CSV file found inside {path}")

        member = csv_members[0]
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
