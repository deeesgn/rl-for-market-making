"""Data ingestion and validation helpers."""

from rl_mm.data.archive import (
    ArchiveError,
    ArchiveInspection,
    detect_archive_type,
    inspect_csv_archive,
)
from rl_mm.data.bybit_downloader import (
    BybitDownloadError,
    BybitDownloadPlan,
    BybitDownloadResult,
    build_download_plan,
    download_file,
    download_plan,
    load_bybit_config,
)
from rl_mm.data.converter import convert_csv_to_parquet, normalize_records
from rl_mm.data.manifest import build_dataset_manifest, write_dataset_manifest
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
    "BybitDownloadPlan",
    "BybitDownloadError",
    "BybitDownloadResult",
    "ORDERBOOK_SCHEMA",
    "TRADES_SCHEMA",
    "SchemaValidationError",
    "build_dataset_manifest",
    "build_download_plan",
    "check_processed_data",
    "convert_csv_to_parquet",
    "detect_archive_type",
    "download_file",
    "download_plan",
    "inspect_csv_archive",
    "load_bybit_config",
    "normalize_records",
    "validate_records",
    "write_dataset_manifest",
]
