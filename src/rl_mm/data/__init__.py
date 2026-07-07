"""Data ingestion and validation helpers."""

from rl_mm.data.archive import (
    ArchiveError,
    ArchiveInspection,
    JsonLinesArchiveInspection,
    detect_archive_type,
    inspect_csv_archive,
    inspect_jsonl_data_archive,
)
from rl_mm.data.bybit_downloader import (
    BybitDownloadError,
    BybitDownloadPlan,
    BybitDownloadResult,
    BybitUrlCandidate,
    BybitUrlProbeResult,
    build_download_plan,
    build_url_candidate,
    build_url_candidates,
    download_file,
    download_plan,
    load_bybit_config,
    probe_url_candidate,
)
from rl_mm.data.converter import convert_csv_to_parquet, normalize_records
from rl_mm.data.manifest import build_dataset_manifest, write_dataset_manifest
from rl_mm.data.orderbook_parser import (
    OrderBookConversionResult,
    convert_orderbook_zip_to_parquet,
    iter_orderbook_messages_from_zip,
    iter_sampled_orderbook_rows,
)
from rl_mm.data.quality import check_processed_data
from rl_mm.data.schema import (
    ORDERBOOK_SCHEMA,
    TRADES_SCHEMA,
    SchemaValidationError,
    validate_records,
)

__all__ = [
    "ArchiveError",
    "ArchiveInspection",
    "JsonLinesArchiveInspection",
    "OrderBookConversionResult",
    "BybitDownloadPlan",
    "BybitDownloadError",
    "BybitDownloadResult",
    "BybitUrlCandidate",
    "BybitUrlProbeResult",
    "ORDERBOOK_SCHEMA",
    "TRADES_SCHEMA",
    "SchemaValidationError",
    "build_dataset_manifest",
    "build_download_plan",
    "build_url_candidate",
    "build_url_candidates",
    "check_processed_data",
    "convert_csv_to_parquet",
    "convert_orderbook_zip_to_parquet",
    "detect_archive_type",
    "download_file",
    "download_plan",
    "inspect_csv_archive",
    "inspect_jsonl_data_archive",
    "iter_orderbook_messages_from_zip",
    "iter_sampled_orderbook_rows",
    "load_bybit_config",
    "normalize_records",
    "probe_url_candidate",
    "validate_records",
    "write_dataset_manifest",
]
