"""Run mock strategy comparisons across simple market regimes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from rl_mm.backtest.metrics import AggregateMetrics
from rl_mm.strategies import BaseStrategy, FixedSpreadStrategy, InventorySkewStrategy
from rl_mm.training import evaluate_ppo

try:
    from scripts.run_baselines import load_config, print_comparison, run_strategy
except ModuleNotFoundError:
    from run_baselines import load_config, print_comparison, run_strategy


DEFAULT_REGIME_CONFIGS = [
    Path("configs/regimes/calm.yaml"),
    Path("configs/regimes/high_volatility.yaml"),
    Path("configs/regimes/trend_up.yaml"),
    Path("configs/regimes/trend_down.yaml"),
    Path("configs/regimes/toxic_flow.yaml"),
]


def load_regime_config(path: Path) -> tuple[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")

    name = str(config.get("name", path.stem))
    overrides = config.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"Expected 'overrides' mapping in {path}")
    return name, overrides


def evaluate_regime(
    *,
    base_config: dict[str, Any],
    regime_overrides: dict[str, Any],
    config_path: Path,
    episodes: int,
    seed: int,
    model_path: Path,
    include_ppo: bool,
) -> dict[str, AggregateMetrics]:
    env_config = {**base_config, **regime_overrides}
    strategies: list[BaseStrategy] = [
        FixedSpreadStrategy(),
        InventorySkewStrategy(),
    ]
    results = {
        strategy.name: run_strategy(
            strategy,
            env_config,
            episodes=episodes,
            seed=seed,
        )
        for strategy in strategies
    }

    if include_ppo:
        temp_config_path = write_temp_regime_config(config_path, env_config)
        try:
            results["ppo"] = evaluate_ppo(
                config_path=temp_config_path,
                model_path=model_path,
                episodes=episodes,
                seed=seed,
            )
        finally:
            temp_config_path.unlink(missing_ok=True)

    return results


def write_temp_regime_config(base_config_path: Path, env_config: dict[str, Any]) -> Path:
    temp_path = base_config_path.with_name(f".{base_config_path.stem}.stress.tmp.yaml")
    with temp_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(env_config, file, sort_keys=True)
    return temp_path


def run_stress(
    *,
    base_config_path: Path,
    regime_paths: list[Path],
    episodes: int,
    seed: int,
    model_path: Path,
) -> dict[str, dict[str, AggregateMetrics]]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    base_config = load_config(base_config_path)
    include_ppo = model_path.is_file()
    if not include_ppo:
        print(
            f"Warning: PPO model not found at {model_path}; "
            "skipping PPO. Run `make train-mock` first."
        )

    results = {}
    for regime_path in regime_paths:
        regime_name, overrides = load_regime_config(regime_path)
        results[regime_name] = evaluate_regime(
            base_config=base_config,
            regime_overrides=overrides,
            config_path=base_config_path,
            episodes=episodes,
            seed=seed,
            model_path=model_path,
            include_ppo=include_ppo,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test mock strategies by market regime.")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base-config", type=Path, default=Path("configs/env_mock.yaml"))
    parser.add_argument("--model-path", type=Path, default=Path("models/mock_ppo.zip"))
    parser.add_argument(
        "--regime-config",
        type=Path,
        action="append",
        dest="regime_configs",
        help="Regime config path. Can be supplied multiple times.",
    )
    args = parser.parse_args()

    regime_paths = args.regime_configs or DEFAULT_REGIME_CONFIGS
    results = run_stress(
        base_config_path=args.base_config,
        regime_paths=regime_paths,
        episodes=args.episodes,
        seed=args.seed,
        model_path=args.model_path,
    )

    for regime_name, regime_results in results.items():
        print()
        print(f"Regime: {regime_name}")
        print_comparison(regime_results)


if __name__ == "__main__":
    main()
