"""Download and convert Bybit quote-saver orderbook archives one day at a time."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse

import requests

from rl_mm.data.orderbook_parser import (
    OrderBookConversionResult,
    convert_orderbook_zip_to_parquet,
)

QUOTE_SAVER_BASE_URL = "https://quote-saver.bycsi.com/orderbook/linear/{symbol}/"
ORDERBOOK_FILENAME_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})_(?P<symbol>[A-Z0-9]+)_ob(?P<depth>\d+)\.data\.zip$"
)
TRANSIENT_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}


class RemoteFileMissingError(RuntimeError):
    """Raised when the remote archive is explicitly not present."""


class TransientDownloadError(RuntimeError):
    """Raised for retryable remote download failures."""


@dataclass(frozen=True)
class DailyOrderBookPipelineResult:
    date: str
    status: str
    url: str
    output_path: str
    raw_zip_size_bytes: int | None
    row_count: int | None
    timestamp_min: str | None
    timestamp_max: str | None
    error: str | None
    processing_seconds: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and convert a Bybit orderbook date range."
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/bybit/orderbook"))
    parser.add_argument("--raw-temp-dir", type=Path, default=Path("data/raw/bybit/tmp/orderbook"))
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--frequency", default="1s")
    parser.add_argument("--keep-raw", action="store_true", default=False)
    parser.add_argument("--max-days", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--read-timeout", type=float, default=300.0)
    parser.add_argument("--retry-backoff-seconds", type=float, default=30.0)
    parser.add_argument("--keep-partial", action="store_true", default=False)
    parser.add_argument("--only-missing", action="store_true", default=False)
    parser.add_argument("--continue-on-error", type=parse_bool, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results = process_orderbook_range(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
        raw_temp_dir=args.raw_temp_dir,
        depth=args.depth,
        frequency=args.frequency,
        keep_raw=args.keep_raw,
        max_days=args.max_days,
        sleep_seconds=args.sleep_seconds,
        retries=args.retries,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        retry_backoff_seconds=args.retry_backoff_seconds,
        keep_partial=args.keep_partial,
        only_missing=args.only_missing,
        continue_on_error=args.continue_on_error,
        dry_run=args.dry_run,
    )
    print_final_summary(
        results=results,
        output_dir=args.output_dir,
        manifest_path=manifest_path(args.output_dir, args.start_date),
    )


def process_orderbook_range(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    raw_temp_dir: Path,
    depth: int,
    frequency: str,
    keep_raw: bool = False,
    max_days: int | None = None,
    sleep_seconds: float = 1.0,
    retries: int = 8,
    connect_timeout: float = 30.0,
    read_timeout: float = 300.0,
    retry_backoff_seconds: float = 30.0,
    keep_partial: bool = False,
    only_missing: bool = False,
    continue_on_error: bool = True,
    dry_run: bool = False,
    download_func: Callable[..., int] | None = None,
    remote_files: list[str] | None = None,
    sleep_func: Callable[[float], None] = time.sleep,
) -> list[DailyOrderBookPipelineResult]:
    requested_dates = list(iter_dates(start_date, end_date))
    if max_days is not None:
        requested_dates = requested_dates[:max_days]
    if only_missing:
        requested_dates = missing_output_dates(
            requested_dates=requested_dates,
            output_dir=output_dir,
            symbol=symbol,
            depth=depth,
            frequency=frequency,
        )
        print("only_missing: true")
        print(f"missing_output_dates_count: {len(requested_dates)}")
        if requested_dates:
            print(
                "missing_output_dates: "
                + ", ".join(current_date.isoformat() for current_date in requested_dates)
            )
    if not requested_dates:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_temp_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_path(output_dir, start_date)
    results = []
    downloader = download_func or download_file
    all_remote_files = (
        sorted(remote_files) if remote_files is not None else fetch_remote_orderbook_files(symbol)
    )
    files_in_range = filter_orderbook_files_by_date(
        all_remote_files,
        requested_dates[0].isoformat(),
        requested_dates[-1].isoformat(),
    )
    files_by_date = map_remote_files_by_date(files_in_range)
    missing_dates = [
        current_date for current_date in requested_dates if current_date not in files_by_date
    ]
    print_coverage_report(
        start_date=requested_dates[0],
        end_date=requested_dates[-1],
        remote_files=all_remote_files,
        files_in_range=files_in_range,
        missing_dates=missing_dates,
    )
    if dry_run:
        print("remote_files_in_range:")
        for filename in files_in_range:
            print(f"  {filename}")

    for index, current_date in enumerate(requested_dates, start=1):
        print(f"[{index}/{len(requested_dates)}] {current_date.isoformat()}")
        remote_filename = files_by_date.get(current_date)
        result = process_orderbook_day(
            symbol=symbol,
            current_date=current_date,
            remote_filename=remote_filename,
            day_index=index,
            total_days=len(requested_dates),
            output_dir=output_dir,
            raw_temp_dir=raw_temp_dir,
            depth=depth,
            frequency=frequency,
            keep_raw=keep_raw,
            retries=retries,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            retry_backoff_seconds=retry_backoff_seconds,
            keep_partial=keep_partial,
            dry_run=dry_run,
            download_func=downloader,
            sleep_func=sleep_func,
        )
        results.append(result)
        if not dry_run:
            append_manifest_entry(manifest, result)
        print_daily_result(result)

        if result.status == "failed" and not continue_on_error:
            raise RuntimeError(result.error or f"Failed processing {current_date}")
        if not dry_run and sleep_seconds > 0 and index < len(requested_dates):
            sleep_func(sleep_seconds)

    return results


def process_orderbook_day(
    *,
    symbol: str,
    current_date: date,
    remote_filename: str | None,
    day_index: int,
    total_days: int,
    output_dir: Path,
    raw_temp_dir: Path,
    depth: int,
    frequency: str,
    keep_raw: bool,
    retries: int,
    connect_timeout: float,
    read_timeout: float,
    retry_backoff_seconds: float,
    keep_partial: bool,
    dry_run: bool,
    download_func: Callable[..., int],
    sleep_func: Callable[[float], None],
) -> DailyOrderBookPipelineResult:
    del day_index, total_days
    started = time.monotonic()
    output_path = output_file_path(
        output_dir=output_dir,
        symbol=symbol,
        current_date=current_date,
        depth=depth,
        frequency=frequency,
    )
    if remote_filename is None:
        return daily_result(
            current_date=current_date,
            status="missing",
            url="",
            output_path=output_path,
            started=started,
            raw_zip_size_bytes=None,
            conversion=None,
            error=None,
        )

    url = build_orderbook_url(symbol=symbol, remote_filename=remote_filename)
    raw_path = raw_file_path(
        raw_temp_dir=raw_temp_dir,
        symbol=symbol,
        remote_filename=remote_filename,
    )
    tmp_output_path = output_path.with_suffix(output_path.suffix + ".tmp")

    if output_path.exists():
        return daily_result(
            current_date=current_date,
            status="skipped",
            url=url,
            output_path=output_path,
            started=started,
            raw_zip_size_bytes=raw_path.stat().st_size if raw_path.exists() else None,
            conversion=None,
            error=None,
        )

    if dry_run:
        print(f"  dry_run remote_filename={remote_filename}")
        print(f"  dry_run url={url}")
        print(f"  output_path={output_path}")
        print(f"  raw_temp_path={raw_path}")
        return daily_result(
            current_date=current_date,
            status="skipped",
            url=url,
            output_path=output_path,
            started=started,
            raw_zip_size_bytes=None,
            conversion=None,
            error=None,
        )

    try:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if tmp_output_path.exists():
            tmp_output_path.unlink()

        raw_zip_size_bytes = download_func(
            url=url,
            output_path=raw_path,
            retries=retries,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            retry_backoff_seconds=retry_backoff_seconds,
            keep_partial=keep_partial,
            sleep_func=sleep_func,
        )
        conversion = convert_orderbook_zip_to_parquet(
            raw_path,
            output_path=tmp_output_path,
            symbol=symbol,
            depth=depth,
            frequency=frequency,
        )
        tmp_output_path.replace(output_path)
        if not keep_raw and raw_path.exists():
            raw_path.unlink()
        return daily_result(
            current_date=current_date,
            status="success",
            url=url,
            output_path=output_path,
            started=started,
            raw_zip_size_bytes=raw_zip_size_bytes,
            conversion=conversion,
            error=None,
        )
    except RemoteFileMissingError as error:
        if tmp_output_path.exists():
            tmp_output_path.unlink()
        if not keep_raw and raw_path.exists():
            raw_path.unlink()
        return daily_result(
            current_date=current_date,
            status="missing",
            url=url,
            output_path=output_path,
            started=started,
            raw_zip_size_bytes=None,
            conversion=None,
            error=str(error),
        )
    except Exception as error:  # noqa: BLE001 - preserve failures in manifest and continue.
        if tmp_output_path.exists():
            tmp_output_path.unlink()
        if not keep_raw and raw_path.exists():
            raw_path.unlink()
        return daily_result(
            current_date=current_date,
            status="failed",
            url=url,
            output_path=output_path,
            started=started,
            raw_zip_size_bytes=raw_path.stat().st_size if raw_path.exists() else None,
            conversion=None,
            error=str(error),
        )


def download_file(
    *,
    url: str,
    output_path: Path,
    retries: int,
    connect_timeout: float = 30.0,
    read_timeout: float = 300.0,
    retry_backoff_seconds: float = 30.0,
    keep_partial: bool = False,
    sleep_func: Callable[[float], None] = time.sleep,
) -> int:
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path.stat().st_size
    delete_partial_download(part_path, keep_partial=False)
    last_error: Exception | None = None
    max_attempts = max(1, retries)
    for attempt in range(1, max_attempts + 1):
        try:
            delete_partial_download(part_path, keep_partial=False)
            with requests.get(
                url,
                stream=True,
                timeout=(connect_timeout, read_timeout),
            ) as response:
                if response.status_code == 404:
                    delete_partial_download(part_path, keep_partial=keep_partial)
                    raise RemoteFileMissingError(f"Remote archive not found: {url} returned 404")
                if response.status_code in TRANSIENT_HTTP_STATUS_CODES:
                    raise TransientDownloadError(
                        f"Remote archive returned retryable HTTP {response.status_code}: {url}"
                    )
                response.raise_for_status()
                with part_path.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
            finalize_download(part_path=part_path, output_path=output_path)
            return output_path.stat().st_size
        except RemoteFileMissingError:
            raise
        except (
            TransientDownloadError,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as error:
            last_error = error
            delete_partial_download(part_path, keep_partial=keep_partial)
            if attempt < max_attempts:
                wait_seconds = retry_backoff_seconds * attempt
                print(
                    f"  retrying download attempt {attempt + 1}/{max_attempts} "
                    f"after {wait_seconds:g}s: {error}"
                )
                sleep_func(wait_seconds)
                continue
        except requests.RequestException as error:
            delete_partial_download(part_path, keep_partial=keep_partial)
            raise RuntimeError(f"Non-retryable download failure for {url}: {error}") from error
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def finalize_download(*, part_path: Path, output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 0 and not part_path.exists():
        return
    if not part_path.exists():
        raise RuntimeError(f"Partial download file is missing: {part_path}")
    if part_path.stat().st_size <= 0:
        raise RuntimeError(f"Partial download file is empty: {part_path}")
    os.replace(part_path, output_path)
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Final download file is missing or empty: {output_path}")


def delete_partial_download(part_path: Path, *, keep_partial: bool) -> None:
    if not keep_partial and part_path.exists():
        part_path.unlink()


def daily_result(
    *,
    current_date: date,
    status: str,
    url: str,
    output_path: Path,
    started: float,
    raw_zip_size_bytes: int | None,
    conversion: OrderBookConversionResult | None,
    error: str | None,
) -> DailyOrderBookPipelineResult:
    return DailyOrderBookPipelineResult(
        date=current_date.isoformat(),
        status=status,
        url=url,
        output_path=str(output_path),
        raw_zip_size_bytes=raw_zip_size_bytes,
        row_count=conversion.row_count if conversion is not None else None,
        timestamp_min=conversion.timestamp_min if conversion is not None else None,
        timestamp_max=conversion.timestamp_max if conversion is not None else None,
        error=error,
        processing_seconds=round(time.monotonic() - started, 3),
    )


def print_daily_result(result: DailyOrderBookPipelineResult) -> None:
    print(
        f"  {result.status}: date={result.date} output={result.output_path} "
        f"raw_size={result.raw_zip_size_bytes} rows={result.row_count} "
        f"seconds={result.processing_seconds}"
    )
    if result.error:
        print(f"  error: {result.error}")


def print_final_summary(
    *,
    results: list[DailyOrderBookPipelineResult],
    output_dir: Path,
    manifest_path: Path,
) -> None:
    parquet_files = sorted(output_dir.rglob("*.parquet")) if output_dir.is_dir() else []
    print("Final summary")
    print(f"total_dates: {len(results)}")
    print(f"successful_dates: {sum(result.status == 'success' for result in results)}")
    print(f"skipped_dates: {sum(result.status == 'skipped' for result in results)}")
    print(f"missing_dates: {sum(result.status == 'missing' for result in results)}")
    print(f"failed_dates: {sum(result.status == 'failed' for result in results)}")
    print(f"total_parquet_files: {len(parquet_files)}")
    print(f"total_processed_size_bytes: {sum(path.stat().st_size for path in parquet_files)}")
    print(f"manifest_path: {manifest_path}")


def append_manifest_entry(path: Path, result: DailyOrderBookPipelineResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(result), sort_keys=True) + "\n")


def fetch_remote_orderbook_files(symbol: str) -> list[str]:
    url = orderbook_directory_url(symbol)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    parser = HrefParser()
    parser.feed(response.text)
    filenames = {
        filename_from_href(href)
        for href in parser.hrefs
        if filename_from_href(href).endswith(".data.zip")
    }
    return sorted(filenames)


def filter_orderbook_files_by_date(
    files: list[str],
    start_date: str,
    end_date: str,
) -> list[str]:
    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    filtered = []
    for filename in files:
        parsed = parse_orderbook_filename(filename)
        if parsed is None:
            continue
        file_date, _, _ = parsed
        if start <= file_date <= end:
            filtered.append(filename)
    return sorted(filtered)


def map_remote_files_by_date(files: list[str]) -> dict[date, str]:
    files_by_date = {}
    for filename in files:
        parsed = parse_orderbook_filename(filename)
        if parsed is None:
            continue
        file_date, _, _ = parsed
        files_by_date.setdefault(file_date, filename)
    return files_by_date


def parse_orderbook_filename(filename: str) -> tuple[date, str, str] | None:
    match = ORDERBOOK_FILENAME_RE.match(filename)
    if match is None:
        return None
    return (
        datetime.fromisoformat(match.group("date")).date(),
        match.group("symbol"),
        f"ob{match.group('depth')}",
    )


def build_orderbook_url(*, symbol: str, remote_filename: str) -> str:
    return f"{orderbook_directory_url(symbol)}{remote_filename}"


def orderbook_directory_url(symbol: str) -> str:
    return QUOTE_SAVER_BASE_URL.format(symbol=symbol.upper())


def filename_from_href(href: str) -> str:
    parsed = urlparse(href)
    return unquote(Path(parsed.path).name)


class HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value:
                self.hrefs.append(value)


def print_coverage_report(
    *,
    start_date: date,
    end_date: date,
    remote_files: list[str],
    files_in_range: list[str],
    missing_dates: list[date],
) -> None:
    remote_dates = [
        parsed[0]
        for filename in files_in_range
        if (parsed := parse_orderbook_filename(filename)) is not None
    ]
    print("Remote orderbook coverage")
    print(f"requested_start_date: {start_date.isoformat()}")
    print(f"requested_end_date: {end_date.isoformat()}")
    print(f"remote_files_found_for_symbol: {len(remote_files)}")
    print(f"remote_files_in_requested_range: {len(files_in_range)}")
    print(f"first_remote_date_in_range: {min(remote_dates).isoformat() if remote_dates else None}")
    print(f"last_remote_date_in_range: {max(remote_dates).isoformat() if remote_dates else None}")
    print(f"missing_requested_dates_count: {len(missing_dates)}")
    if missing_dates:
        print(
            "missing_requested_dates: "
            + ", ".join(current_date.isoformat() for current_date in missing_dates)
        )


def output_file_path(
    *,
    output_dir: Path,
    symbol: str,
    current_date: date,
    depth: int,
    frequency: str,
) -> Path:
    safe_frequency = frequency.replace("/", "_")
    filename = (
        f"{symbol.upper()}_{current_date.isoformat()}_orderbook_top"
        f"{depth}_{safe_frequency}.parquet"
    )
    return output_dir / symbol.upper() / filename


def raw_file_path(*, raw_temp_dir: Path, symbol: str, remote_filename: str) -> Path:
    return raw_temp_dir / symbol.upper() / remote_filename


def missing_output_dates(
    *,
    requested_dates: list[date],
    output_dir: Path,
    symbol: str,
    depth: int,
    frequency: str,
) -> list[date]:
    return [
        current_date
        for current_date in requested_dates
        if not output_file_path(
            output_dir=output_dir,
            symbol=symbol,
            current_date=current_date,
            depth=depth,
            frequency=frequency,
        ).exists()
    ]


def manifest_path(output_dir: Path, start_date: str) -> Path:
    year = datetime.fromisoformat(start_date).date().year
    return output_dir / f"manifest_orderbook_{year}.jsonl"


def iter_dates(start_date: str, end_date: str):
    current = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    if end < current:
        raise ValueError("end_date must be on or after start_date")
    while current <= end:
        yield current
        current += timedelta(days=1)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


if __name__ == "__main__":
    main()
