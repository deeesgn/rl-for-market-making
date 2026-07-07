import json
from pathlib import Path

import pandas as pd

from rl_mm.data.manifest import build_dataset_manifest, infer_dataset_type, write_dataset_manifest


def test_build_dataset_manifest_scans_parquet_files(tmp_path: Path) -> None:
    trades_dir = tmp_path / "trades"
    trades_dir.mkdir()
    parquet_path = trades_dir / "sample.parquet"
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01T00:00:00+00:00",
                "symbol": "BTCUSDT",
                "side": "buy",
                "price": 42000.0,
                "size": 0.1,
            }
        ]
    ).to_parquet(parquet_path, index=False)

    manifest = build_dataset_manifest(tmp_path)

    assert manifest["input_dir"] == str(tmp_path)
    assert len(manifest["files"]) == 1
    entry = manifest["files"][0]
    assert entry["file_path"] == str(parquet_path)
    assert entry["dataset_type"] == "trades"
    assert entry["row_count"] == 1
    assert entry["columns"] == ("timestamp", "symbol", "side", "price", "size")
    assert entry["timestamp_min"] == "2024-01-01T00:00:00+00:00"
    assert entry["timestamp_max"] == "2024-01-01T00:00:00+00:00"
    assert entry["file_size_bytes"] > 0


def test_write_dataset_manifest(tmp_path: Path) -> None:
    input_dir = tmp_path / "input" / "orderbook"
    output_path = tmp_path / "manifest.json"
    input_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01T00:00:00+00:00",
                "symbol": "BTCUSDT",
                "bid_price": 42000.0,
                "bid_size": 1.0,
                "ask_price": 42001.0,
                "ask_size": 1.2,
            }
        ]
    ).to_parquet(input_dir / "book.parquet", index=False)

    write_dataset_manifest(tmp_path / "input", output_path)

    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["files"][0]["dataset_type"] == "orderbook"


def test_infer_dataset_type_unknown() -> None:
    assert infer_dataset_type(Path("data/processed/bybit/unknown/file.parquet")) is None
