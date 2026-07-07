"""Project configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


class ConfigValidationError(ValueError):
    """Raised when a project config is missing required fields."""


@dataclass(frozen=True)
class ExperimentProtocol:
    """Validated project-wide experiment protocol."""

    exchange: str
    instrument: str
    start_date: date
    end_date: date
    train_split: float
    validation_split: float
    test_split: float
    split_method: str
    shuffle: bool
    replay_frequency: str
    book_depth: int
    initial_capital: float
    maker_fee: float
    taker_fee: float
    latency_ms: float
    max_inventory_btc: float
    seeds: tuple[int, ...]
    evaluation_episodes: int
    required_metrics: tuple[str, ...]

    @property
    def split_sum(self) -> float:
        return self.train_split + self.validation_split + self.test_split


REQUIRED_PROTOCOL_METRICS = (
    "total_pnl",
    "total_reward",
    "sharpe_ratio",
    "maximum_drawdown",
    "max_abs_inventory",
    "mean_abs_inventory",
    "final_inventory",
    "quote_rate",
    "fill_rate",
)


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ConfigValidationError(f"Expected YAML mapping in {path}")
    return config


def load_experiment_protocol(
    path: Path = Path("configs/experiment_protocol.yaml"),
) -> ExperimentProtocol:
    """Load and validate the project-wide experiment protocol."""

    config = load_yaml_config(path)
    required_top_level = {
        "exchange",
        "instrument",
        "date_range",
        "split",
        "data",
        "trading",
        "evaluation",
    }
    missing = required_top_level - set(config)
    if missing:
        raise ConfigValidationError(f"Experiment protocol missing fields: {sorted(missing)}")

    date_range = require_mapping(config, "date_range")
    split = require_mapping(config, "split")
    data = require_mapping(config, "data")
    trading = require_mapping(config, "trading")
    evaluation = require_mapping(config, "evaluation")
    require_fields(date_range, "date_range", {"start", "end"})
    require_fields(split, "split", {"train", "validation", "test", "method", "shuffle"})
    require_fields(data, "data", {"replay_frequency", "book_depth"})
    require_fields(
        trading,
        "trading",
        {
            "initial_capital",
            "maker_fee",
            "taker_fee",
            "latency_ms",
            "max_inventory_btc",
        },
    )
    require_fields(evaluation, "evaluation", {"seeds", "episodes", "required_metrics"})

    protocol = ExperimentProtocol(
        exchange=str(config["exchange"]),
        instrument=str(config["instrument"]),
        start_date=parse_date(date_range["start"]),
        end_date=parse_date(date_range["end"]),
        train_split=float(split["train"]),
        validation_split=float(split["validation"]),
        test_split=float(split["test"]),
        split_method=str(split["method"]),
        shuffle=bool(split["shuffle"]),
        replay_frequency=str(data["replay_frequency"]),
        book_depth=int(data["book_depth"]),
        initial_capital=parse_numeric_field(trading, "initial_capital"),
        maker_fee=parse_numeric_field(trading, "maker_fee"),
        taker_fee=parse_numeric_field(trading, "taker_fee"),
        latency_ms=parse_numeric_field(trading, "latency_ms"),
        max_inventory_btc=parse_numeric_field(trading, "max_inventory_btc"),
        seeds=tuple(int(seed) for seed in evaluation["seeds"]),
        evaluation_episodes=int(evaluation["episodes"]),
        required_metrics=tuple(str(metric) for metric in evaluation["required_metrics"]),
    )
    validate_experiment_protocol(protocol)
    return protocol


def validate_experiment_protocol(protocol: ExperimentProtocol) -> None:
    """Validate protocol invariants that future scripts rely on."""

    if protocol.end_date < protocol.start_date:
        raise ConfigValidationError("Protocol end date must be on or after start date.")
    if protocol.split_method != "chronological":
        raise ConfigValidationError("Protocol split method must be chronological.")
    if protocol.shuffle:
        raise ConfigValidationError("Protocol split must not shuffle data.")
    if abs(protocol.split_sum - 1.0) > 1e-9:
        raise ConfigValidationError("Train/validation/test split must sum to 1.0.")
    if protocol.book_depth <= 0:
        raise ConfigValidationError("Book depth must be positive.")
    if protocol.initial_capital <= 0:
        raise ConfigValidationError("Initial capital must be positive.")
    if protocol.maker_fee < 0:
        raise ConfigValidationError("Maker fee must be non-negative.")
    if protocol.taker_fee < 0:
        raise ConfigValidationError("Taker fee must be non-negative.")
    if protocol.latency_ms < 0:
        raise ConfigValidationError("Latency must be non-negative.")
    if protocol.max_inventory_btc <= 0:
        raise ConfigValidationError("Max inventory must be positive.")
    if protocol.evaluation_episodes <= 0:
        raise ConfigValidationError("Evaluation episodes must be positive.")

    missing_metrics = set(REQUIRED_PROTOCOL_METRICS) - set(protocol.required_metrics)
    if missing_metrics:
        raise ConfigValidationError(f"Protocol missing required metrics: {sorted(missing_metrics)}")


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config[key]
    if not isinstance(value, dict):
        raise ConfigValidationError(f"Expected '{key}' to be a mapping.")
    return value


def require_fields(config: dict[str, Any], name: str, fields: set[str]) -> None:
    missing = fields - set(config)
    if missing:
        raise ConfigValidationError(f"'{name}' missing fields: {sorted(missing)}")


def parse_date(value: Any) -> date:
    return date.fromisoformat(str(value))


def parse_numeric_field(config: dict[str, Any], key: str) -> float:
    if key not in config:
        raise ConfigValidationError(f"Missing numeric field '{key}'.")
    try:
        return float(config[key])
    except (TypeError, ValueError) as error:
        raise ConfigValidationError(f"Field '{key}' must be numeric.") from error
