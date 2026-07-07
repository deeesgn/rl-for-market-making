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


@dataclass(frozen=True)
class BybitUrlCandidate:
    name: str
    dataset: str
    symbol: str
    date: date
    filename: str
    url: str


@dataclass(frozen=True)
class BybitUrlProbeResult:
    candidate: BybitUrlCandidate
    method: str
    status_code: int | None
    content_type: str | None
    content_length: str | None
    working: bool
    error: str | None = None


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
    template_name: str | None = None,
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
            if template_name is not None:
                candidate = build_url_candidate(
                    config,
                    dataset=str(dataset),
                    symbol=symbol,
                    date_value=current_date,
                    template_name=template_name,
                )
                filename = candidate.filename
                url = candidate.url
            else:
                filename_template = filename_templates.get(
                    dataset,
                    "{symbol}_{dataset}_{date}.csv.gz",
                )
                filename = format_bybit_template(
                    filename_template,
                    symbol=symbol,
                    dataset=str(dataset),
                    date_value=current_date,
                    filename="",
                )
                url = (
                    format_bybit_template(
                        template,
                        symbol=symbol,
                        dataset=str(dataset),
                        date_value=current_date,
                        filename=filename,
                    )
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


def build_url_candidates(
    config: dict[str, Any],
    *,
    dataset: str,
    symbol: str,
    date_value: date | str,
) -> list[BybitUrlCandidate]:
    get_schema(dataset)
    current_date = parse_date_value(date_value)
    candidates = []
    for name, candidate_config in iter_candidate_template_configs(config, dataset=dataset):
        candidates.append(
            build_url_candidate_from_config(
                name=name,
                candidate_config=candidate_config,
                dataset=dataset,
                symbol=symbol,
                date_value=current_date,
            )
        )
    return candidates


def build_url_candidate(
    config: dict[str, Any],
    *,
    dataset: str,
    symbol: str,
    date_value: date | str,
    template_name: str,
) -> BybitUrlCandidate:
    current_date = parse_date_value(date_value)
    for name, candidate_config in iter_candidate_template_configs(config, dataset=dataset):
        if name == template_name:
            return build_url_candidate_from_config(
                name=name,
                candidate_config=candidate_config,
                dataset=dataset,
                symbol=symbol,
                date_value=current_date,
            )
    known = ", ".join(name for name, _ in iter_candidate_template_configs(config, dataset=dataset))
    raise ValueError(
        f"Unknown template_name '{template_name}' for {dataset}. "
        f"Known templates: {known or '<none>'}"
    )


def iter_candidate_template_configs(config: dict[str, Any], *, dataset: str):
    candidates_by_dataset = config.get("candidate_url_templates", {})
    candidates = candidates_by_dataset.get(dataset)
    if candidates is None:
        url_template = config.get("url_templates", {}).get(dataset, "")
        filename_template = config.get("filename_templates", {}).get(
            dataset,
            "{symbol}_{dataset}_{date}.csv.gz",
        )
        if url_template:
            yield "default", {
                "filename_template": filename_template,
                "url_template": url_template,
            }
        return

    if isinstance(candidates, dict):
        for name, candidate_config in candidates.items():
            yield str(name), candidate_config
        return

    for index, candidate_config in enumerate(candidates, start=1):
        name = str(candidate_config.get("name", f"candidate_{index}"))
        yield name, candidate_config


def build_url_candidate_from_config(
    *,
    name: str,
    candidate_config: dict[str, Any],
    dataset: str,
    symbol: str,
    date_value: date,
) -> BybitUrlCandidate:
    symbol = symbol.upper()
    filename_template = candidate_config["filename_template"]
    url_template = candidate_config["url_template"]
    filename = format_bybit_template(
        filename_template,
        symbol=symbol,
        dataset=dataset,
        date_value=date_value,
        filename="",
    )
    url = format_bybit_template(
        url_template,
        symbol=symbol,
        dataset=dataset,
        date_value=date_value,
        filename=filename,
    )
    return BybitUrlCandidate(
        name=name,
        dataset=dataset,
        symbol=symbol,
        date=date_value,
        filename=filename,
        url=url,
    )


def format_bybit_template(
    template: str,
    *,
    symbol: str,
    dataset: str,
    date_value: date,
    filename: str,
) -> str:
    date_text = date_value.isoformat()
    return template.format(
        symbol=symbol,
        symbol_lower=symbol.lower(),
        dataset=dataset,
        date=date_text,
        date_compact=date_text.replace("-", ""),
        filename=filename,
    )


def parse_date_value(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(value).date()


def iter_dates(start_date: str, end_date: str):
    current = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    if end < current:
        raise ValueError("end_date must be on or after start_date")
    while current <= end:
        yield current
        current += timedelta(days=1)


def probe_url_candidate(
    candidate: BybitUrlCandidate,
    *,
    timeout: float = 10.0,
    request_head: Any | None = None,
    request_get: Any | None = None,
) -> BybitUrlProbeResult:
    head = request_head or requests.head
    get = request_get or requests.get

    try:
        response = head(candidate.url, timeout=timeout, allow_redirects=True)
        if response.status_code >= 400:
            return probe_url_candidate_with_get(candidate, timeout=timeout, request_get=get)
        return build_probe_result(candidate, method="HEAD", response=response)
    except requests.RequestException as error:
        fallback_result = probe_url_candidate_with_get(
            candidate,
            timeout=timeout,
            request_get=get,
        )
        if fallback_result.error:
            return fallback_result
        return BybitUrlProbeResult(
            candidate=fallback_result.candidate,
            method=fallback_result.method,
            status_code=fallback_result.status_code,
            content_type=fallback_result.content_type,
            content_length=fallback_result.content_length,
            working=fallback_result.working,
            error=f"HEAD failed: {error}",
        )


def probe_url_candidate_with_get(
    candidate: BybitUrlCandidate,
    *,
    timeout: float,
    request_get: Any,
) -> BybitUrlProbeResult:
    try:
        response = request_get(candidate.url, timeout=timeout, stream=True)
        try:
            return build_probe_result(candidate, method="GET", response=response)
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()
    except requests.RequestException as error:
        return BybitUrlProbeResult(
            candidate=candidate,
            method="GET",
            status_code=None,
            content_type=None,
            content_length=None,
            working=False,
            error=str(error),
        )


def build_probe_result(
    candidate: BybitUrlCandidate,
    *,
    method: str,
    response: Any,
) -> BybitUrlProbeResult:
    status_code = int(response.status_code)
    headers = response.headers
    return BybitUrlProbeResult(
        candidate=candidate,
        method=method,
        status_code=status_code,
        content_type=headers.get("content-type"),
        content_length=headers.get("content-length"),
        working=200 <= status_code < 400,
    )


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
