"""Probe candidate Bybit archive URLs without downloading full files."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_mm.data.bybit_downloader import (
    build_url_candidates,
    load_bybit_config,
    probe_url_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover working Bybit archive URL templates.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_bybit.yaml"))
    parser.add_argument("--dataset", choices=["trades", "orderbook"], required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--date")
    parser.add_argument("--dates")
    parser.add_argument("--timeout", type=float)
    args = parser.parse_args()
    if (args.date is None) == (args.dates is None):
        parser.error("provide exactly one of --date or --dates")

    config = load_bybit_config(args.config)
    timeout = args.timeout or float(config.get("timeout_seconds", 30))

    dates = parse_dates(args.date, args.dates)
    for index, date_value in enumerate(dates):
        if index:
            print()
        discover_date(
            config,
            dataset=args.dataset,
            symbol=args.symbol,
            date_value=date_value,
            timeout=timeout,
        )


def parse_dates(date_value: str | None, dates_value: str | None) -> list[str]:
    if date_value is not None:
        return [date_value]
    if dates_value is None:
        return []
    return [value.strip() for value in dates_value.split(",") if value.strip()]


def discover_date(
    config: dict,
    *,
    dataset: str,
    symbol: str,
    date_value: str,
    timeout: float,
) -> None:
    candidates = build_url_candidates(
        config,
        dataset=dataset,
        symbol=symbol,
        date_value=date_value,
    )

    print("Bybit URL discovery")
    print(f"dataset: {dataset}")
    print(f"symbol: {symbol.upper()}")
    print(f"date: {date_value}")
    print("template | method | status | working | content_type | content_length | url")
    print("-" * 100)

    working_count = 0
    for candidate in candidates:
        result = probe_url_candidate(candidate, timeout=timeout)
        if result.working:
            working_count += 1
        status = result.status_code if result.status_code is not None else "error"
        content_type = result.content_type or "-"
        content_length = result.content_length or "-"
        working = "YES" if result.working else "no"
        print(
            f"{candidate.name} | {result.method} | {status} | {working} | "
            f"{content_type} | {content_length} | {candidate.url}"
        )
        if result.error:
            print(f"  error: {result.error}")

    print(f"working_candidates: {working_count}/{len(candidates)}")


if __name__ == "__main__":
    main()
