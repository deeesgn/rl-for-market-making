from pathlib import Path

import pytest
import requests

from rl_mm.data.bybit_downloader import (
    BybitDownloadError,
    build_download_plan,
    download_file,
    download_plan,
)


class FakeResponse:
    def __init__(self, content: bytes = b"archive-data", status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError("bad status", response=self)


def test_build_download_plan_supports_cli_overrides_and_max_files(tmp_path: Path) -> None:
    plan = build_download_plan(
        {
            "symbol": "ETHUSDT",
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "datasets": ["trades", "orderbook"],
            "raw_dir": tmp_path / "raw",
            "processed_dir": tmp_path / "processed",
            "url_templates": {"trades": "https://example.test/{symbol}/{filename}"},
        },
        dataset="trades",
        symbol="btcusdt",
        start_date="2024-01-02",
        end_date="2024-01-05",
        max_files=2,
    )

    assert len(plan) == 2
    assert plan[0].dataset == "trades"
    assert plan[0].symbol == "BTCUSDT"
    assert plan[0].raw_path == tmp_path / "raw/trades/BTCUSDT/BTCUSDT_trades_2024-01-02.csv.gz"
    assert plan[0].url == "https://example.test/BTCUSDT/BTCUSDT_trades_2024-01-02.csv.gz"


def test_download_file_writes_archive_with_mocked_request(tmp_path: Path) -> None:
    plan = build_download_plan(
        {
            "symbol": "BTCUSDT",
            "start_date": "2024-01-01",
            "end_date": "2024-01-01",
            "datasets": ["trades"],
            "raw_dir": tmp_path / "raw",
            "processed_dir": tmp_path / "processed",
            "url_templates": {"trades": "https://example.test/{symbol}/{filename}"},
        }
    )[0]
    calls = []

    def fake_get(url: str, *, timeout: float) -> FakeResponse:
        calls.append((url, timeout))
        return FakeResponse(content=b"downloaded")

    result = download_file(plan, timeout=7.5, request_get=fake_get)

    assert result.status == "downloaded"
    assert result.bytes_written == len(b"downloaded")
    assert plan.raw_path.read_bytes() == b"downloaded"
    assert calls == [(plan.url, 7.5)]


def test_download_file_skips_existing_file_without_network(tmp_path: Path) -> None:
    plan = build_download_plan(
        {
            "symbol": "BTCUSDT",
            "start_date": "2024-01-01",
            "end_date": "2024-01-01",
            "datasets": ["trades"],
            "raw_dir": tmp_path / "raw",
            "processed_dir": tmp_path / "processed",
            "url_templates": {"trades": "https://example.test/{symbol}/{filename}"},
        }
    )[0]
    plan.raw_path.parent.mkdir(parents=True)
    plan.raw_path.write_bytes(b"existing")

    def fail_get(url: str, *, timeout: float) -> FakeResponse:
        raise AssertionError(f"unexpected network call to {url} with timeout {timeout}")

    result = download_file(plan, request_get=fail_get)

    assert result.status == "skipped_existing"
    assert result.bytes_written == len(b"existing")


def test_download_plan_reports_http_errors_clearly(tmp_path: Path) -> None:
    plan = build_download_plan(
        {
            "symbol": "BTCUSDT",
            "start_date": "2024-01-01",
            "end_date": "2024-01-01",
            "datasets": ["trades"],
            "raw_dir": tmp_path / "raw",
            "processed_dir": tmp_path / "processed",
            "url_templates": {"trades": "https://example.test/{symbol}/{filename}"},
        }
    )

    def fake_get(url: str, *, timeout: float) -> FakeResponse:
        return FakeResponse(status_code=404)

    with pytest.raises(BybitDownloadError, match="HTTP error.*status=404"):
        download_plan(plan, request_get=fake_get)
