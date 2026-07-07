"""Bybit historical data download planning scaffold.

This module intentionally avoids real network download logic for now. It builds
deterministic download plans, validates local files, and defines the storage
layout that later hftbacktest integration can consume.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from rl_mm.data.schema import get_schema, validate_records


@dataclass(frozen=True)
class BybitDownloadPlan:
    dataset: str
    symbol: str
    date: date
    url: str
    raw_path: Path
    processed_path: Path


def load_bybit_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config


def build_download_plan(config: dict[str, Any]) -> list[BybitDownloadPlan]:
    symbol = str(config["symbol"])
    raw_dir = Path(config.get("raw_dir", "data/raw/bybit"))
    processed_dir = Path(config.get("processed_dir", "data/processed/bybit"))
    datasets = config.get("datasets", ["trades", "orderbook"])
    url_templates = config.get("url_templates", {})
    dates = list(iter_dates(str(config["start_date"]), str(config["end_date"])))

    plans = []
    for dataset in datasets:
        get_schema(str(dataset))
        template = url_templates.get(dataset, "")
        for current_date in dates:
            date_text = current_date.isoformat()
            filename = f"{symbol}_{dataset}_{date_text}.csv.gz"
            url = (
                template.format(symbol=symbol, date=date_text, filename=filename)
                if template
                else ""
            )
            plans.append(
                BybitDownloadPlan(
                    dataset=str(dataset),
                    symbol=symbol,
                    date=current_date,
                    url=url,
                    raw_path=raw_dir / str(dataset) / filename,
                    processed_path=processed_dir / str(dataset) / f"{symbol}_{date_text}.parquet",
                )
            )
    return plans


def iter_dates(start_date: str, end_date: str):
    current = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    if end < current:
        raise ValueError("end_date must be on or after start_date")
    while current <= end:
        yield current
        current += timedelta(days=1)


def print_dry_run(plan: list[BybitDownloadPlan]) -> None:
    print("Bybit download dry run")
    for item in plan:
        print(
            f"- {item.dataset} {item.symbol} {item.date}: "
            f"url={item.url or '<configured later>'} "
            f"raw={item.raw_path} processed={item.processed_path}"
        )


def validate_local_csv(path: Path, *, dataset: str) -> None:
    schema = get_schema(dataset)
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        validate_records(reader, schema)
