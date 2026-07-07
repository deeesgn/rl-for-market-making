from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from scripts.download_convert_orderbook_range import (
    build_orderbook_url,
    manifest_path,
    output_file_path,
    process_orderbook_range,
)


def test_orderbook_range_dry_run_does_not_download_or_write(tmp_path: Path) -> None:
    calls = []

    def fake_download(url: str, output_path: Path, retries: int) -> int:
        calls.append((url, output_path, retries))
        return 0

    results = process_orderbook_range(
        symbol="BTCUSDT",
        start_date="2025-01-01",
        end_date="2025-01-02",
        output_dir=tmp_path / "processed",
        raw_temp_dir=tmp_path / "raw",
        depth=10,
        frequency="1s",
        dry_run=True,
        download_func=fake_download,
        sleep_func=lambda seconds: None,
    )

    assert [result.status for result in results] == ["skipped", "skipped"]
    assert calls == []
    assert not manifest_path(tmp_path / "processed", "2025-01-01").exists()


def test_orderbook_range_downloads_converts_deletes_raw_and_writes_manifest(
    tmp_path: Path,
) -> None:
    source_zip = write_orderbook_zip(tmp_path / "source.zip")
    output_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"

    def fake_download(url: str, output_path: Path, retries: int) -> int:
        output_path.write_bytes(source_zip.read_bytes())
        return output_path.stat().st_size

    results = process_orderbook_range(
        symbol="BTCUSDT",
        start_date="2025-01-01",
        end_date="2025-01-01",
        output_dir=output_dir,
        raw_temp_dir=raw_dir,
        depth=2,
        frequency="1s",
        retries=2,
        dry_run=False,
        download_func=fake_download,
        sleep_func=lambda seconds: None,
    )

    output_path = output_file_path(
        output_dir=output_dir,
        symbol="BTCUSDT",
        current_date=pd.Timestamp("2025-01-01").date(),
        depth=2,
        frequency="1s",
    )
    manifest = manifest_path(output_dir, "2025-01-01")

    assert results[0].status == "success"
    assert results[0].row_count == 1
    assert output_path.is_file()
    assert not list(raw_dir.rglob("*.zip"))
    assert not output_path.with_suffix(output_path.suffix + ".tmp").exists()

    manifest_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert manifest_rows[0]["status"] == "success"
    assert manifest_rows[0]["row_count"] == 1
    assert manifest_rows[0]["error"] is None


def test_orderbook_range_skips_existing_parquet(tmp_path: Path) -> None:
    output_dir = tmp_path / "processed"
    output_path = output_file_path(
        output_dir=output_dir,
        symbol="BTCUSDT",
        current_date=pd.Timestamp("2025-01-01").date(),
        depth=10,
        frequency="1s",
    )
    output_path.parent.mkdir(parents=True)
    pd.DataFrame([{"timestamp": "2025-01-01T00:00:00+00:00"}]).to_parquet(
        output_path,
        index=False,
    )

    def fail_download(url: str, output_path: Path, retries: int) -> int:
        raise AssertionError(f"unexpected download for {url}")

    results = process_orderbook_range(
        symbol="BTCUSDT",
        start_date="2025-01-01",
        end_date="2025-01-01",
        output_dir=output_dir,
        raw_temp_dir=tmp_path / "raw",
        depth=10,
        frequency="1s",
        dry_run=False,
        download_func=fail_download,
        sleep_func=lambda seconds: None,
    )

    assert results[0].status == "skipped"
    assert results[0].output_path == str(output_path)


def test_build_orderbook_url_uses_quote_saver_pattern() -> None:
    url = build_orderbook_url(
        symbol="btcusdt",
        current_date=pd.Timestamp("2025-05-01").date(),
    )

    assert url == (
        "https://quote-saver.bycsi.com/orderbook/linear/"
        "BTCUSDT/2025-05-01_BTCUSDT_ob500.data.zip"
    )


def write_orderbook_zip(path: Path) -> Path:
    message = {
        "topic": "orderbook.500.BTCUSDT",
        "type": "snapshot",
        "ts": 1_735_689_600_000,
        "cts": 1_735_689_600_000,
        "data": {
            "s": "BTCUSDT",
            "b": [["100", "1"], ["99", "2"]],
            "a": [["101", "1"], ["102", "2"]],
            "u": 1,
            "seq": 2,
        },
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("2025-01-01_BTCUSDT_ob500.data", json.dumps(message))
    return path
