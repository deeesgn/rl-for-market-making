"""Download and aggregate daily Bybit public trades to one-second parquet."""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

TRADE_COLUMNS = [
    "timestamp",
    "buy_volume",
    "sell_volume",
    "max_buy_price",
    "min_sell_price",
]
BYBIT_TRADES_URL = "https://public.bybit.com/trading/{symbol}/{symbol}{date}.csv.gz"


def aggregate_trade_chunks(chunks: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate normalized or raw Bybit trade chunks into UTC one-second buckets."""

    aggregates = []
    for chunk in chunks:
        required = {"timestamp", "side", "size", "price"}
        missing = required.difference(chunk.columns)
        if missing:
            raise ValueError("Missing trade columns: " + ", ".join(sorted(missing)))
        frame = chunk.loc[:, ["timestamp", "side", "size", "price"]].copy()
        numeric_timestamp = pd.to_numeric(frame["timestamp"], errors="coerce")
        frame["timestamp"] = pd.to_datetime(numeric_timestamp, unit="s", utc=True).dt.floor("s")
        frame["size"] = pd.to_numeric(frame["size"], errors="coerce")
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        frame["side"] = frame["side"].astype(str).str.lower()
        frame = frame.dropna(subset=["timestamp", "size", "price"])
        buy = frame["side"].eq("buy")
        sell = frame["side"].eq("sell")
        frame["buy_volume"] = frame["size"].where(buy, 0.0)
        frame["sell_volume"] = frame["size"].where(sell, 0.0)
        frame["max_buy_price"] = frame["price"].where(buy)
        frame["min_sell_price"] = frame["price"].where(sell)
        aggregates.append(
            frame.groupby("timestamp", as_index=False).agg(
                buy_volume=("buy_volume", "sum"),
                sell_volume=("sell_volume", "sum"),
                max_buy_price=("max_buy_price", "max"),
                min_sell_price=("min_sell_price", "min"),
            )
        )

    if not aggregates:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    combined = pd.concat(aggregates, ignore_index=True)
    result = combined.groupby("timestamp", as_index=False).agg(
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
        max_buy_price=("max_buy_price", "max"),
        min_sell_price=("min_sell_price", "min"),
    )
    return result.loc[:, TRADE_COLUMNS].sort_values("timestamp").reset_index(drop=True)


def aggregate_trade_archive(path: Path, *, chunksize: int = 500_000) -> pd.DataFrame:
    chunks = pd.read_csv(
        path,
        usecols=["timestamp", "side", "size", "price"],
        chunksize=chunksize,
        compression="infer",
    )
    return aggregate_trade_chunks(chunks)


def prepare_trade_range(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    raw_temp_dir: Path,
    retries: int = 5,
) -> list[tuple[str, str]]:
    """Download, aggregate, and clean up one daily archive at a time."""

    symbol = symbol.upper()
    output_dir = output_dir / symbol
    raw_temp_dir = raw_temp_dir / symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_temp_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for current_date in iter_dates(start_date, end_date):
        day = current_date.isoformat()
        output_path = output_dir / f"{symbol}_{day}_trades_1s.parquet"
        if output_path.is_file():
            print(f"trades {day}: skipped")
            results.append((day, "skipped"))
            continue
        raw_path = raw_temp_dir / f"{symbol}{day}.csv.gz"
        url = BYBIT_TRADES_URL.format(symbol=symbol, date=day)
        try:
            download_trade_archive(url, raw_path, retries=retries)
            aggregated = aggregate_trade_archive(raw_path)
            temp_output = output_path.with_suffix(".tmp.parquet")
            aggregated.to_parquet(temp_output, index=False)
            os.replace(temp_output, output_path)
            raw_path.unlink(missing_ok=True)
            print(f"trades {day}: success rows={len(aggregated)}")
            results.append((day, "success"))
        except Exception:
            raw_path.with_suffix(raw_path.suffix + ".part").unlink(missing_ok=True)
            raise
    return results


def download_trade_archive(url: str, output_path: Path, *, retries: int) -> int:
    if output_path.is_file() and output_path.stat().st_size > 0:
        return output_path.stat().st_size
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    part_path.unlink(missing_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with part_path.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
            if not part_path.is_file() or part_path.stat().st_size == 0:
                raise RuntimeError(f"Downloaded trade archive is empty: {url}")
            os.replace(part_path, output_path)
            return output_path.stat().st_size
        except requests.RequestException as error:
            last_error = error
            part_path.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(5 * attempt)
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def iter_dates(start_date: str, end_date: str):
    current = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < current:
        raise ValueError("end_date must be on or after start_date")
    while current <= end:
        yield current
        current += timedelta(days=1)
