"""Download and convert Bybit quote-saver orderbook archives one day at a time."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import requests

from rl_mm.data.orderbook_parser import (
    OrderBookConversionResult,
    convert_orderbook_zip_to_parquet,
)

QUOTE_SAVER_URL_TEMPLATE = (
    "https://quote-saver.bycsi.com/orderbook/linear/"
    "{symbol}/{date}_{symbol}_ob500.data.zip"
)


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
    parser.add_argument("--retries", type=int, default=3)
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
    retries: int = 3,
    continue_on_error: bool = True,
    dry_run: bool = False,
    download_func: Callable[[str, Path, int], int] | None = None,
    sleep_func: Callable[[float], None] = time.sleep,
) -> list[DailyOrderBookPipelineResult]:
    dates = list(iter_dates(start_date, end_date))
    if max_days is not None:
        dates = dates[:max_days]

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_temp_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_path(output_dir, start_date)
    results = []
    downloader = download_func or download_file

    for index, current_date in enumerate(dates, start=1):
        print(f"[{index}/{len(dates)}] {current_date.isoformat()}")
        result = process_orderbook_day(
            symbol=symbol,
            current_date=current_date,
            day_index=index,
            total_days=len(dates),
            output_dir=output_dir,
            raw_temp_dir=raw_temp_dir,
            depth=depth,
            frequency=frequency,
            keep_raw=keep_raw,
            retries=retries,
            dry_run=dry_run,
            download_func=downloader,
        )
        results.append(result)
        if not dry_run:
            append_manifest_entry(manifest, result)
        print_daily_result(result)

        if result.status == "failed" and not continue_on_error:
            raise RuntimeError(result.error or f"Failed processing {current_date}")
        if not dry_run and sleep_seconds > 0 and index < len(dates):
            sleep_func(sleep_seconds)

    return results


def process_orderbook_day(
    *,
    symbol: str,
    current_date: date,
    day_index: int,
    total_days: int,
    output_dir: Path,
    raw_temp_dir: Path,
    depth: int,
    frequency: str,
    keep_raw: bool,
    retries: int,
    dry_run: bool,
    download_func: Callable[[str, Path, int], int],
) -> DailyOrderBookPipelineResult:
    del day_index, total_days
    started = time.monotonic()
    url = build_orderbook_url(symbol=symbol, current_date=current_date)
    output_path = output_file_path(
        output_dir=output_dir,
        symbol=symbol,
        current_date=current_date,
        depth=depth,
        frequency=frequency,
    )
    raw_path = raw_file_path(raw_temp_dir=raw_temp_dir, symbol=symbol, current_date=current_date)
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

        raw_zip_size_bytes = download_func(url, raw_path, retries)
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
    except Exception as error:  # noqa: BLE001 - preserve failures in manifest and continue.
        if tmp_output_path.exists():
            tmp_output_path.unlink()
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


def download_file(url: str, output_path: Path, retries: int) -> int:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()
                with output_path.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
            return output_path.stat().st_size
        except requests.RequestException as error:
            last_error = error
            if output_path.exists():
                output_path.unlink()
            if attempt < retries:
                time.sleep(min(attempt, 5))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


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
    print(f"failed_dates: {sum(result.status == 'failed' for result in results)}")
    print(f"total_parquet_files: {len(parquet_files)}")
    print(f"total_processed_size_bytes: {sum(path.stat().st_size for path in parquet_files)}")
    print(f"manifest_path: {manifest_path}")


def append_manifest_entry(path: Path, result: DailyOrderBookPipelineResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(asdict(result), sort_keys=True) + "\n")


def build_orderbook_url(*, symbol: str, current_date: date) -> str:
    return QUOTE_SAVER_URL_TEMPLATE.format(
        symbol=symbol.upper(),
        date=current_date.isoformat(),
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


def raw_file_path(*, raw_temp_dir: Path, symbol: str, current_date: date) -> Path:
    return (
        raw_temp_dir
        / symbol.upper()
        / f"{current_date.isoformat()}_{symbol.upper()}_ob500.data.zip"
    )


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
