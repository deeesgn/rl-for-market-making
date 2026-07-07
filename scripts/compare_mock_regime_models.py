"""Compare all mock strategies and PPO models across all regimes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rl_mm.backtest.metrics import AggregateMetrics
from rl_mm.strategies import BaseStrategy, FixedSpreadStrategy, InventorySkewStrategy
from rl_mm.training import evaluate_ppo

try:
    from scripts.run_baselines import load_config, print_comparison, run_strategy
    from scripts.run_mock_regime_stress import (
        DEFAULT_REGIME_CONFIGS,
        load_regime_config,
        write_temp_regime_config,
    )
except ModuleNotFoundError:
    from run_baselines import load_config, print_comparison, run_strategy
    from run_mock_regime_stress import (
        DEFAULT_REGIME_CONFIGS,
        load_regime_config,
        write_temp_regime_config,
    )


def warn_missing_model(label: str, model_path: Path) -> bool:
    """Return whether the model exists, warning clearly when it does not."""

    if model_path.is_file():
        return True

    print(f"Warning: {label} model not found at {model_path}; skipping {label}.")
    return False


def compare_regime_models(
    *,
    base_config: dict[str, Any],
    regime_overrides: dict[str, Any],
    base_config_path: Path,
    episodes: int,
    seed: int,
    static_model_path: Path,
    randomized_model_path: Path,
    include_static_ppo: bool,
    include_randomized_ppo: bool,
) -> dict[str, AggregateMetrics]:
    """Evaluate baselines plus available PPO models for one regime."""

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

    if include_static_ppo or include_randomized_ppo:
        temp_config_path = write_temp_regime_config(base_config_path, env_config)
        try:
            if include_static_ppo:
                results["static_ppo"] = evaluate_ppo(
                    config_path=temp_config_path,
                    model_path=static_model_path,
                    episodes=episodes,
                    seed=seed,
                )
            if include_randomized_ppo:
                results["randomized_ppo"] = evaluate_ppo(
                    config_path=temp_config_path,
                    model_path=randomized_model_path,
                    episodes=episodes,
                    seed=seed,
                )
        finally:
            temp_config_path.unlink(missing_ok=True)

    return results


def compare_all_regimes(
    *,
    base_config_path: Path,
    regime_paths: list[Path],
    episodes: int,
    seed: int,
    static_model_path: Path,
    randomized_model_path: Path,
) -> dict[str, dict[str, AggregateMetrics]]:
    """Evaluate all configured regimes with the same seed schedule."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    base_config = load_config(base_config_path)
    include_static_ppo = warn_missing_model("static_ppo", static_model_path)
    include_randomized_ppo = warn_missing_model("randomized_ppo", randomized_model_path)

    results = {}
    for regime_path in regime_paths:
        regime_name, overrides = load_regime_config(regime_path)
        results[regime_name] = compare_regime_models(
            base_config=base_config,
            regime_overrides=overrides,
            base_config_path=base_config_path,
            episodes=episodes,
            seed=seed,
            static_model_path=static_model_path,
            randomized_model_path=randomized_model_path,
            include_static_ppo=include_static_ppo,
            include_randomized_ppo=include_randomized_ppo,
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare mock baselines, static PPO, and randomized PPO across regimes."
    )
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base-config", type=Path, default=Path("configs/env_mock.yaml"))
    parser.add_argument("--static-model-path", type=Path, default=Path("models/mock_ppo.zip"))
    parser.add_argument(
        "--randomized-model-path",
        type=Path,
        default=Path("models/mock_ppo_randomized.zip"),
    )
    args = parser.parse_args()

    results = compare_all_regimes(
        base_config_path=args.base_config,
        regime_paths=DEFAULT_REGIME_CONFIGS,
        episodes=args.episodes,
        seed=args.seed,
        static_model_path=args.static_model_path,
        randomized_model_path=args.randomized_model_path,
    )

    for regime_name, regime_results in results.items():
        print()
        print(f"Regime: {regime_name}")
        print_comparison(regime_results)


if __name__ == "__main__":
    main()
