"""Compare mock rule-based strategies and a trained PPO model."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_mm.backtest.metrics import AggregateMetrics
from rl_mm.strategies import BaseStrategy, FixedSpreadStrategy, InventorySkewStrategy
from rl_mm.training import evaluate_ppo

try:
    from scripts.run_baselines import load_config, print_comparison, run_strategy
except ModuleNotFoundError:
    from run_baselines import load_config, print_comparison, run_strategy


def ensure_model_exists(model_path: Path) -> None:
    """Fail with an actionable message when the PPO model is missing."""

    if not model_path.is_file():
        raise FileNotFoundError(
            f"PPO model not found at {model_path}. Run `make train-mock` first."
        )


def compare_strategies(
    *,
    config_path: Path,
    episodes: int,
    seed: int,
    model_path: Path,
) -> dict[str, AggregateMetrics]:
    """Evaluate all mock strategies over the same episode seeds."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    ensure_model_exists(model_path)
    env_config = load_config(config_path)
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
    results["ppo"] = evaluate_ppo(
        config_path=config_path,
        model_path=model_path,
        episodes=episodes,
        seed=seed,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed-spread, inventory-skew, and PPO on the mock env."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/env_mock.yaml"))
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("models/mock_ppo.zip"))
    args = parser.parse_args()

    try:
        results = compare_strategies(
            config_path=args.config,
            episodes=args.episodes,
            seed=args.seed,
            model_path=args.model_path,
        )
    except FileNotFoundError as error:
        parser.exit(status=1, message=f"{error}\n")

    print_comparison(results)


if __name__ == "__main__":
    main()
