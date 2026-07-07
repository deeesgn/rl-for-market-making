"""Bybit historical data download planning and execution helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests
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


@dataclass(frozen=True)
class BybitDownloadResult:
    plan: BybitDownloadPlan
    status: str
    bytes_written: int


class BybitDownloadError(RuntimeError):
    """Raised when a Bybit archive request fails."""


def load_bybit_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config


def build_download_plan(
    config: dict[str, Any],
    *,
    dataset: str | None = None,
    symbol: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    max_files: int | None = None,
) -> list[BybitDownloadPlan]:
    if max_files is not None and max_files < 0:
        raise ValueError("max_files must be non-negative")
    if max_files == 0:
        return []

    symbol = str(symbol or config["symbol"]).upper()
    raw_dir = Path(config.get("raw_dir", "data/raw/bybit"))
    processed_dir = Path(config.get("processed_dir", "data/processed/bybit"))
    datasets = [dataset] if dataset else config.get("datasets", ["trades", "orderbook"])
    filename_templates = config.get("filename_templates", {})
    url_templates = config.get("url_templates", {})
    dates = list(
        iter_dates(
            str(start_date or config["start_date"]),
            str(end_date or config["end_date"]),
        )
    )

    plans = []
    for dataset in datasets:
        get_schema(str(dataset))
        template = url_templates.get(dataset, "")
        for current_date in dates:
            date_text = current_date.isoformat()
            filename_template = filename_templates.get(
                dataset,
                "{symbol}_{dataset}_{date}.csv.gz",
            )
            filename = filename_template.format(
                symbol=symbol,
                dataset=dataset,
                date=date_text,
            )
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
                    raw_path=raw_dir / str(dataset) / symbol / filename,
                    processed_path=processed_dir
                    / str(dataset)
                    / symbol
                    / f"{symbol}_{date_text}.parquet",
                )
            )
            if max_files is not None and len(plans) >= max_files:
                return plans
    return plans


def iter_dates(start_date: str, end_date: str):
    current = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    if end < current:
        raise ValueError("end_date must be on or after start_date")
    while current <= end:
        yield current
        current += timedelta(days=1)


def download_file(
    plan: BybitDownloadPlan,
    *,
    timeout: float = 30.0,
    request_get: Any | None = None,
) -> BybitDownloadResult:
    if plan.raw_path.exists():
        return BybitDownloadResult(
            plan=plan,
            status="skipped_existing",
            bytes_written=plan.raw_path.stat().st_size,
        )
    if not plan.url:
        raise BybitDownloadError(
            f"Missing URL template for {plan.dataset}. Add it to configs/data_bybit.yaml."
        )

    plan.raw_path.parent.mkdir(parents=True, exist_ok=True)
    getter = request_get or requests.get

    try:
        response = getter(plan.url, timeout=timeout)
        response.raise_for_status()
    except requests.HTTPError as error:
        status_code = getattr(error.response, "status_code", "unknown")
        raise BybitDownloadError(
            f"HTTP error while downloading {plan.url}: status={status_code}"
        ) from error
    except requests.RequestException as error:
        raise BybitDownloadError(f"Request failed while downloading {plan.url}: {error}") from error

    content = response.content
    plan.raw_path.write_bytes(content)
    return BybitDownloadResult(
        plan=plan,
        status="downloaded",
        bytes_written=len(content),
    )


def download_plan(
    plan: list[BybitDownloadPlan],
    *,
    timeout: float = 30.0,
    request_get: Any | None = None,
) -> list[BybitDownloadResult]:
    return [
        download_file(item, timeout=timeout, request_get=request_get)
        for item in plan
    ]


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
