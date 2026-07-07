from __future__ import annotations

from datetime import date

import requests

from rl_mm.data.bybit_downloader import (
    BybitUrlCandidate,
    build_url_candidates,
    probe_url_candidate,
)


class FakeProbeResponse:
    def __init__(
        self,
        status_code: int,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_build_url_candidates_uses_named_config_templates() -> None:
    candidates = build_url_candidates(
        {
            "candidate_url_templates": {
                "trades": {
                    "zip_candidate": {
                        "filename_template": "{symbol}-{date}.zip",
                        "url_template": "https://example.test/{symbol}/{filename}",
                    },
                    "compact_candidate": {
                        "filename_template": "{symbol}{date_compact}.csv.gz",
                        "url_template": "https://mirror.test/{symbol_lower}/{filename}",
                    },
                }
            }
        },
        dataset="trades",
        symbol="btcusdt",
        date_value="2024-01-01",
    )

    assert [candidate.name for candidate in candidates] == [
        "zip_candidate",
        "compact_candidate",
    ]
    assert candidates[0].url == "https://example.test/BTCUSDT/BTCUSDT-2024-01-01.zip"
    assert candidates[1].url == "https://mirror.test/btcusdt/BTCUSDT20240101.csv.gz"


def test_probe_url_candidate_uses_head_first() -> None:
    candidate = BybitUrlCandidate(
        name="ok",
        dataset="trades",
        symbol="BTCUSDT",
        date=date(2024, 1, 1),
        filename="sample.csv.gz",
        url="https://example.test/sample.csv.gz",
    )
    calls = []

    def fake_head(url: str, *, timeout: float, allow_redirects: bool) -> FakeProbeResponse:
        calls.append(("HEAD", url, timeout, allow_redirects))
        return FakeProbeResponse(
            200,
            headers={"content-type": "text/csv", "content-length": "123"},
        )

    def fake_get(url: str, *, timeout: float, stream: bool) -> FakeProbeResponse:
        raise AssertionError(f"unexpected GET for {url} {timeout} {stream}")

    result = probe_url_candidate(
        candidate,
        timeout=5,
        request_head=fake_head,
        request_get=fake_get,
    )

    assert result.method == "HEAD"
    assert result.status_code == 200
    assert result.working is True
    assert result.content_type == "text/csv"
    assert result.content_length == "123"
    assert calls == [("HEAD", candidate.url, 5, True)]


def test_probe_url_candidate_falls_back_to_streaming_get_when_head_not_supported() -> None:
    candidate = BybitUrlCandidate(
        name="head_405",
        dataset="trades",
        symbol="BTCUSDT",
        date=date(2024, 1, 1),
        filename="sample.csv.gz",
        url="https://example.test/sample.csv.gz",
    )
    get_response = FakeProbeResponse(
        200,
        headers={"content-type": "application/gzip", "content-length": "456"},
    )
    calls = []

    def fake_head(url: str, *, timeout: float, allow_redirects: bool) -> FakeProbeResponse:
        calls.append(("HEAD", url, timeout, allow_redirects))
        return FakeProbeResponse(405)

    def fake_get(url: str, *, timeout: float, stream: bool) -> FakeProbeResponse:
        calls.append(("GET", url, timeout, stream))
        return get_response

    result = probe_url_candidate(
        candidate,
        timeout=8,
        request_head=fake_head,
        request_get=fake_get,
    )

    assert result.method == "GET"
    assert result.status_code == 200
    assert result.working is True
    assert result.content_type == "application/gzip"
    assert result.content_length == "456"
    assert get_response.closed is True
    assert calls == [
        ("HEAD", candidate.url, 8, True),
        ("GET", candidate.url, 8, True),
    ]


def test_probe_url_candidate_reports_request_errors() -> None:
    candidate = BybitUrlCandidate(
        name="broken",
        dataset="trades",
        symbol="BTCUSDT",
        date=date(2024, 1, 1),
        filename="sample.csv.gz",
        url="https://example.test/sample.csv.gz",
    )

    def fake_head(url: str, *, timeout: float, allow_redirects: bool) -> FakeProbeResponse:
        raise requests.ConnectionError("connection failed")

    result = probe_url_candidate(candidate, request_head=fake_head)

    assert result.method == "HEAD"
    assert result.status_code is None
    assert result.working is False
    assert result.error == "connection failed"
