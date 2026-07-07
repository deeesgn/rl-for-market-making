from pathlib import Path

import pytest
import yaml

from rl_mm.config import (
    REQUIRED_PROTOCOL_METRICS,
    ConfigValidationError,
    load_experiment_protocol,
)


def test_experiment_protocol_loads_expected_values() -> None:
    protocol = load_experiment_protocol(Path("configs/experiment_protocol.yaml"))

    assert protocol.exchange == "Bybit"
    assert protocol.instrument == "BTCUSDT Perpetual"
    assert protocol.start_date.isoformat() == "2024-01-01"
    assert protocol.end_date.isoformat() == "2025-12-31"
    assert protocol.train_split == 0.75
    assert protocol.validation_split == 0.10
    assert protocol.test_split == 0.15
    assert protocol.replay_frequency == "1s"
    assert protocol.book_depth == 10
    assert protocol.initial_capital == 10000
    assert protocol.maker_fee == 0.0002
    assert protocol.taker_fee == 0.00055
    assert protocol.latency_ms == 0
    assert protocol.max_inventory_btc == 0.02
    assert protocol.seeds == (42, 100, 200)
    assert protocol.evaluation_episodes == 20
    assert protocol.required_metrics == REQUIRED_PROTOCOL_METRICS


def test_experiment_protocol_splits_sum_to_one() -> None:
    protocol = load_experiment_protocol(Path("configs/experiment_protocol.yaml"))

    assert protocol.split_sum == pytest.approx(1.0)


def test_experiment_protocol_rejects_invalid_split(tmp_path: Path) -> None:
    config_path = tmp_path / "protocol.yaml"
    config = {
        "exchange": "Bybit",
        "instrument": "BTCUSDT Perpetual",
        "date_range": {"start": "2024-01-01", "end": "2025-12-31"},
        "split": {"train": 0.7, "validation": 0.1, "test": 0.1},
        "data": {"replay_frequency": "1s", "book_depth": 10},
        "trading": {
            "initial_capital": 10000,
            "maker_fee": 0.0002,
            "taker_fee": 0.00055,
            "latency_ms": 0,
            "max_inventory_btc": 0.02,
        },
        "evaluation": {"seeds": [42, 100, 200], "episodes": 20},
        "required_metrics": list(REQUIRED_PROTOCOL_METRICS),
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="split must sum"):
        load_experiment_protocol(config_path)


def test_experiment_protocol_requires_numeric_trading_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "protocol.yaml"
    config = {
        "exchange": "Bybit",
        "instrument": "BTCUSDT Perpetual",
        "date_range": {"start": "2024-01-01", "end": "2025-12-31"},
        "split": {"train": 0.75, "validation": 0.1, "test": 0.15},
        "data": {"replay_frequency": "1s", "book_depth": 10},
        "trading": {
            "initial_capital": 10000,
            "maker_fee": "not-a-number",
            "taker_fee": 0.00055,
            "latency_ms": 0,
            "max_inventory_btc": 0.02,
        },
        "evaluation": {"seeds": [42, 100, 200], "episodes": 20},
        "required_metrics": list(REQUIRED_PROTOCOL_METRICS),
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="maker_fee"):
        load_experiment_protocol(config_path)
