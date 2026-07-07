from __future__ import annotations

import gzip
import sys
import zipfile
from pathlib import Path

import pytest

from rl_mm.data.archive import ArchiveError, detect_archive_type, inspect_csv_archive
from scripts.verify_bybit_archive import main as verify_archive_main

CSV_TEXT = "\n".join(
    [
        "timestamp,price,size",
        "2024-01-01T00:00:00Z,42000.0,0.1",
        "2024-01-01T00:00:01Z,42001.0,0.2",
    ]
)


def test_inspect_plain_csv_archive(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    path.write_text(CSV_TEXT, encoding="utf-8")

    inspection = inspect_csv_archive(path)

    assert inspection.archive_type == "csv"
    assert inspection.read_path == path
    assert inspection.row_count == 2
    assert inspection.columns == ["timestamp", "price", "size"]
    assert inspection.first_rows[0]["price"] == "42000.0"


def test_inspect_gzip_csv_archive(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as file:
        file.write(CSV_TEXT)

    inspection = inspect_csv_archive(path)

    assert inspection.archive_type == "csv.gz"
    assert inspection.read_path == path
    assert inspection.row_count == 2
    assert inspection.columns == ["timestamp", "price", "size"]


def test_inspect_zip_archive_extracts_first_csv(tmp_path: Path) -> None:
    path = tmp_path / "sample.zip"
    extract_dir = tmp_path / "extracted"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("nested/sample.csv", CSV_TEXT)

    inspection = inspect_csv_archive(path, extract_dir=extract_dir)

    assert inspection.archive_type == "zip"
    assert inspection.read_path == extract_dir / "sample.csv"
    assert inspection.row_count == 2
    assert inspection.columns == ["timestamp", "price", "size"]


def test_zip_without_csv_raises_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "sample.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("notes.txt", "not csv")

    with pytest.raises(ArchiveError, match="No CSV file"):
        inspect_csv_archive(path)


def test_detect_archive_type_rejects_unknown_extension(tmp_path: Path) -> None:
    with pytest.raises(ArchiveError, match="Unsupported archive type"):
        detect_archive_type(tmp_path / "sample.json")


def test_verify_bybit_archive_dry_run_prints_url_and_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "data_bybit.yaml"
    config_path.write_text(
        "\n".join(
            [
                "symbol: BTCUSDT",
                'start_date: "2024-01-01"',
                'end_date: "2024-01-01"',
                "datasets:",
                "  - trades",
                f"raw_dir: {tmp_path / 'raw'}",
                f"processed_dir: {tmp_path / 'processed'}",
                "filename_templates:",
                '  trades: "{symbol}-{date}.zip"',
                "url_templates:",
                '  trades: "https://example.test/{symbol}/{filename}"',
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_bybit_archive.py",
            "--config",
            str(config_path),
            "--dataset",
            "trades",
            "--symbol",
            "BTCUSDT",
            "--date",
            "2024-01-01",
            "--output-dir",
            str(tmp_path / "verify"),
        ],
    )

    verify_archive_main()

    output = capsys.readouterr().out
    assert "Bybit archive verification dry run" in output
    assert "https://example.test/BTCUSDT/BTCUSDT-2024-01-01.zip" in output
    assert str(tmp_path / "verify/trades/BTCUSDT/BTCUSDT-2024-01-01.zip") in output
