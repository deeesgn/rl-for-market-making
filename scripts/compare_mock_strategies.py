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


def print_action_distribution(results: dict[str, AggregateMetrics]) -> None:
    """Print mean action counts per strategy."""

    action_columns = [
        "action_0_no_quote",
        "action_1_narrow",
        "action_2_medium",
        "action_3_wide",
        "action_4_skew_sell",
        "action_5_skew_buy",
    ]
    columns = ["strategy", "action_0", "action_1", "action_2", "action_3", "action_4", "action_5"]
    rows = []
    for strategy_name, metrics in results.items():
        metric_row = metrics.as_dict()
        rows.append(
            {
                "strategy": strategy_name,
                **{
                    column: f"{metric_row[action_column]['mean']:.1f}"
                    for column, action_column in zip(columns[1:], action_columns, strict=True)
                },
            }
        )

    widths = {
        column: max(len(column), *(len(row[column]) for row in rows))
        for column in columns
    }
    print()
    print("Action distribution, mean count per episode")
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare fixed-spread, inventory-skew, and PPO on the mock env."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/env_mock.yaml"))
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("models/mock_ppo.zip"))
    parser.add_argument("--show-actions", action="store_true")
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
    if args.show_actions:
        print_action_distribution(results)


if __name__ == "__main__":
    main()
