from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import requests

from scripts.download_convert_orderbook_range import (
    build_orderbook_url,
    download_file,
    fetch_remote_orderbook_files,
    filter_orderbook_files_by_date,
    finalize_download,
    manifest_path,
    output_file_path,
    parse_orderbook_filename,
    process_orderbook_range,
)


def test_orderbook_range_dry_run_does_not_download_or_write(tmp_path: Path) -> None:
    calls = []

    def fake_download(**kwargs) -> int:
        calls.append(kwargs)
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
        remote_files=["2025-01-01_BTCUSDT_ob200.data.zip"],
        sleep_func=lambda seconds: None,
    )

    assert [result.status for result in results] == ["skipped", "missing"]
    assert calls == []
    assert not manifest_path(tmp_path / "processed", "2025-01-01").exists()


def test_orderbook_range_downloads_converts_deletes_raw_and_writes_manifest(
    tmp_path: Path,
) -> None:
    source_zip = write_orderbook_zip(tmp_path / "source.zip")
    output_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"

    def fake_download(**kwargs) -> int:
        output_path = kwargs["output_path"]
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
        remote_files=["2025-01-01_BTCUSDT_ob200.data.zip"],
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

    def fail_download(**kwargs) -> int:
        raise AssertionError(f"unexpected download for {kwargs['url']}")

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
        remote_files=["2025-01-01_BTCUSDT_ob500.data.zip"],
        sleep_func=lambda seconds: None,
    )

    assert results[0].status == "skipped"
    assert results[0].output_path == str(output_path)


def test_only_missing_mode_processes_only_missing_dates(tmp_path: Path) -> None:
    source_zip = write_orderbook_zip(tmp_path / "source.zip")
    output_dir = tmp_path / "processed"
    existing_path = output_file_path(
        output_dir=output_dir,
        symbol="BTCUSDT",
        current_date=pd.Timestamp("2025-01-01").date(),
        depth=2,
        frequency="1s",
    )
    existing_path.parent.mkdir(parents=True)
    existing_path.write_bytes(b"existing parquet")
    calls = []

    def fake_download(**kwargs) -> int:
        calls.append(kwargs["url"])
        kwargs["output_path"].write_bytes(source_zip.read_bytes())
        return kwargs["output_path"].stat().st_size

    results = process_orderbook_range(
        symbol="BTCUSDT",
        start_date="2025-01-01",
        end_date="2025-01-03",
        output_dir=output_dir,
        raw_temp_dir=tmp_path / "raw",
        depth=2,
        frequency="1s",
        only_missing=True,
        download_func=fake_download,
        remote_files=[
            "2025-01-01_BTCUSDT_ob500.data.zip",
            "2025-01-03_BTCUSDT_ob200.data.zip",
        ],
        sleep_func=lambda seconds: None,
    )

    assert [result.date for result in results] == ["2025-01-02", "2025-01-03"]
    assert [result.status for result in results] == ["missing", "success"]
    assert calls == [
        "https://quote-saver.bycsi.com/orderbook/linear/"
        "BTCUSDT/2025-01-03_BTCUSDT_ob200.data.zip"
    ]
    assert existing_path.read_bytes() == b"existing parquet"


def test_only_missing_mode_never_overwrites_existing_parquet(tmp_path: Path) -> None:
    output_dir = tmp_path / "processed"
    output_path = output_file_path(
        output_dir=output_dir,
        symbol="BTCUSDT",
        current_date=pd.Timestamp("2025-01-01").date(),
        depth=10,
        frequency="1s",
    )
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"do not touch")

    def fail_download(**kwargs) -> int:
        raise AssertionError(f"unexpected download for {kwargs['url']}")

    results = process_orderbook_range(
        symbol="BTCUSDT",
        start_date="2025-01-01",
        end_date="2025-01-01",
        output_dir=output_dir,
        raw_temp_dir=tmp_path / "raw",
        depth=10,
        frequency="1s",
        only_missing=True,
        download_func=fail_download,
        remote_files=["2025-01-01_BTCUSDT_ob500.data.zip"],
        sleep_func=lambda seconds: None,
    )

    assert results == []
    assert output_path.read_bytes() == b"do not touch"


def test_fetch_remote_orderbook_files_parses_directory_listing(
    monkeypatch,
) -> None:
    class FakeResponse:
        text = """
        <html>
          <a href="2025-05-01_BTCUSDT_ob200.data.zip">ob200</a>
          <a href="/orderbook/linear/BTCUSDT/2025-05-02_BTCUSDT_ob500.data.zip">ob500</a>
          <a href="notes.txt">notes</a>
        </html>
        """

        def raise_for_status(self) -> None:
            return None

    calls = []

    def fake_get(url: str, *, timeout: float) -> FakeResponse:
        calls.append((url, timeout))
        return FakeResponse()

    monkeypatch.setattr(
        "scripts.download_convert_orderbook_range.requests.get",
        fake_get,
    )

    files = fetch_remote_orderbook_files("btcusdt")

    assert files == [
        "2025-05-01_BTCUSDT_ob200.data.zip",
        "2025-05-02_BTCUSDT_ob500.data.zip",
    ]
    assert calls == [("https://quote-saver.bycsi.com/orderbook/linear/BTCUSDT/", 30)]


def test_filter_orderbook_files_by_date_keeps_exact_ob200_and_ob500_names() -> None:
    files = [
        "2025-04-30_BTCUSDT_ob500.data.zip",
        "2025-05-01_BTCUSDT_ob200.data.zip",
        "2025-05-02_BTCUSDT_ob500.data.zip",
        "2025-05-03_BTCUSDT_ob200.data.zip",
        "README.txt",
    ]

    assert filter_orderbook_files_by_date(files, "2025-05-01", "2025-05-02") == [
        "2025-05-01_BTCUSDT_ob200.data.zip",
        "2025-05-02_BTCUSDT_ob500.data.zip",
    ]
    assert parse_orderbook_filename("2025-05-01_BTCUSDT_ob200.data.zip") == (
        pd.Timestamp("2025-05-01").date(),
        "BTCUSDT",
        "ob200",
    )
    assert parse_orderbook_filename("2025-05-02_BTCUSDT_ob500.data.zip") == (
        pd.Timestamp("2025-05-02").date(),
        "BTCUSDT",
        "ob500",
    )


def test_missing_dates_do_not_trigger_guessed_urls(tmp_path: Path) -> None:
    source_zip = write_orderbook_zip(tmp_path / "source.zip")
    calls = []

    def fake_download(**kwargs) -> int:
        url = kwargs["url"]
        output_path = kwargs["output_path"]
        calls.append(url)
        output_path.write_bytes(source_zip.read_bytes())
        return output_path.stat().st_size

    results = process_orderbook_range(
        symbol="BTCUSDT",
        start_date="2025-05-01",
        end_date="2025-05-03",
        output_dir=tmp_path / "processed",
        raw_temp_dir=tmp_path / "raw",
        depth=2,
        frequency="1s",
        dry_run=False,
        download_func=fake_download,
        remote_files=["2025-05-01_BTCUSDT_ob200.data.zip"],
        sleep_func=lambda seconds: None,
    )

    assert [result.status for result in results] == ["success", "missing", "missing"]
    assert calls == [
        "https://quote-saver.bycsi.com/orderbook/linear/"
        "BTCUSDT/2025-05-01_BTCUSDT_ob200.data.zip"
    ]


def test_build_orderbook_url_uses_exact_remote_filename() -> None:
    url = build_orderbook_url(
        symbol="btcusdt",
        remote_filename="2025-08-21_BTCUSDT_ob200.data.zip",
    )

    assert url == (
        "https://quote-saver.bycsi.com/orderbook/linear/"
        "BTCUSDT/2025-08-21_BTCUSDT_ob200.data.zip"
    )


def test_download_file_retries_transient_timeout(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_get(url: str, *, stream: bool, timeout: tuple[float, float]):
        calls.append((url, stream, timeout))
        if len(calls) == 1:
            raise requests.ReadTimeout("temporary slow server")
        return FakeStreamResponse(chunks=[b"zip-bytes"])

    monkeypatch.setattr(
        "scripts.download_convert_orderbook_range.requests.get",
        fake_get,
    )

    output_path = tmp_path / "archive.zip"
    size = download_file(
        url="https://example.test/archive.zip",
        output_path=output_path,
        retries=2,
        connect_timeout=30,
        read_timeout=300,
        retry_backoff_seconds=0,
        sleep_func=lambda seconds: None,
    )

    assert size == len(b"zip-bytes")
    assert output_path.read_bytes() == b"zip-bytes"
    assert len(calls) == 2
    assert calls[1] == ("https://example.test/archive.zip", True, (30, 300))


def test_download_file_does_not_retry_404(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_get(url: str, *, stream: bool, timeout: tuple[float, float]):
        calls.append((url, stream, timeout))
        return FakeStreamResponse(status_code=404)

    monkeypatch.setattr(
        "scripts.download_convert_orderbook_range.requests.get",
        fake_get,
    )

    output_path = tmp_path / "missing.zip"

    try:
        download_file(
            url="https://example.test/missing.zip",
            output_path=output_path,
            retries=8,
            retry_backoff_seconds=0,
            sleep_func=lambda seconds: None,
        )
    except RuntimeError as error:
        assert "404" in str(error)
    else:
        raise AssertionError("expected 404 download failure")

    assert len(calls) == 1
    assert not output_path.exists()
    assert not output_path.with_suffix(".zip.part").exists()


def test_download_file_uses_part_file_and_renames_on_success(
    monkeypatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "archive.zip"
    part_path = output_path.with_suffix(".zip.part")

    def fake_get(url: str, *, stream: bool, timeout: tuple[float, float]):
        return FakeStreamResponse(
            chunks=[b"a", b"b"],
            on_iter=lambda: assert_path_exists(part_path),
        )

    monkeypatch.setattr(
        "scripts.download_convert_orderbook_range.requests.get",
        fake_get,
    )

    size = download_file(
        url="https://example.test/archive.zip",
        output_path=output_path,
        retries=1,
        retry_backoff_seconds=0,
        sleep_func=lambda seconds: None,
    )

    assert size == 2
    assert output_path.read_bytes() == b"ab"
    assert not part_path.exists()


def test_failed_download_does_not_create_final_zip(monkeypatch, tmp_path: Path) -> None:
    def fake_get(url: str, *, stream: bool, timeout: tuple[float, float]):
        raise requests.ConnectionError("connection reset by peer")

    monkeypatch.setattr(
        "scripts.download_convert_orderbook_range.requests.get",
        fake_get,
    )

    output_path = tmp_path / "archive.zip"

    try:
        download_file(
            url="https://example.test/archive.zip",
            output_path=output_path,
            retries=1,
            retry_backoff_seconds=0,
            sleep_func=lambda seconds: None,
        )
    except RuntimeError as error:
        assert "connection reset by peer" in str(error)
    else:
        raise AssertionError("expected failed download")

    assert not output_path.exists()
    assert not output_path.with_suffix(".zip.part").exists()


def test_download_file_removes_stale_part_before_retry(monkeypatch, tmp_path: Path) -> None:
    output_path = tmp_path / "archive.zip"
    part_path = output_path.with_suffix(".zip.part")
    part_path.write_bytes(b"stale partial")

    def fake_get(url: str, *, stream: bool, timeout: tuple[float, float]):
        return FakeStreamResponse(
            chunks=[b"fresh"],
            on_iter=lambda: assert_path_bytes(part_path, b""),
        )

    monkeypatch.setattr(
        "scripts.download_convert_orderbook_range.requests.get",
        fake_get,
    )

    size = download_file(
        url="https://example.test/archive.zip",
        output_path=output_path,
        retries=1,
        retry_backoff_seconds=0,
        sleep_func=lambda seconds: None,
    )

    assert size == len(b"fresh")
    assert output_path.read_bytes() == b"fresh"
    assert not part_path.exists()


def test_download_file_reuses_existing_final_zip(monkeypatch, tmp_path: Path) -> None:
    output_path = tmp_path / "archive.zip"
    output_path.write_bytes(b"existing")

    def fail_get(url: str, *, stream: bool, timeout: tuple[float, float]):
        raise AssertionError(f"unexpected network call for {url}")

    monkeypatch.setattr(
        "scripts.download_convert_orderbook_range.requests.get",
        fail_get,
    )

    size = download_file(
        url="https://example.test/archive.zip",
        output_path=output_path,
        retries=1,
        retry_backoff_seconds=0,
        sleep_func=lambda seconds: None,
    )

    assert size == len(b"existing")
    assert output_path.read_bytes() == b"existing"


def test_finalize_download_accepts_existing_final_when_part_is_missing(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "archive.zip"
    part_path = output_path.with_suffix(".zip.part")
    output_path.write_bytes(b"already renamed")

    finalize_download(part_path=part_path, output_path=output_path)

    assert output_path.read_bytes() == b"already renamed"


def assert_path_exists(path: Path) -> None:
    assert path.exists()


def assert_path_bytes(path: Path, expected: bytes) -> None:
    assert path.read_bytes() == expected


class FakeStreamResponse:
    def __init__(
        self,
        *,
        chunks: list[bytes] | None = None,
        status_code: int = 200,
        on_iter=None,
    ) -> None:
        self.chunks = chunks or []
        self.status_code = status_code
        self.on_iter = on_iter

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int):
        del chunk_size
        if self.on_iter is not None:
            self.on_iter()
        yield from self.chunks


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
