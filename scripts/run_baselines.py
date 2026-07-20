"""Run rule-based baselines in the mock market-making environment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from rl_mm.backtest.metrics import (
    AggregateMetrics,
    EpisodeMetrics,
    aggregate_episode_metrics,
    compute_episode_metrics,
)
from rl_mm.env import MockMarketMakingEnv
from rl_mm.strategies import BaseStrategy, FixedSpreadStrategy, InventorySkewStrategy


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    config.pop("actions", None)
    return config


def run_strategy_episode(
    strategy: BaseStrategy,
    env_config: dict[str, Any],
    *,
    seed: int,
) -> EpisodeMetrics:
    episode_config = {**env_config, "seed": seed}
    env = MockMarketMakingEnv(**episode_config)
    observation, _ = env.reset()

    rewards: list[float] = []
    pnls: list[float] = []
    inventories = [float(observation["inventory"])]
    quoted: list[bool] = []
    actions: list[int] = []
    bid_fills: list[bool] = []
    ask_fills: list[bool] = []

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = strategy.select_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        actions.append(int(action))
        rewards.append(float(reward))
        pnls.append(float(info["pnl"]))
        inventories.append(float(observation["inventory"]))
        quoted.append(bool(info["quoted"]))
        bid_fills.append(bool(info["bid_filled"]))
        ask_fills.append(bool(info["ask_filled"]))

    return compute_episode_metrics(
        rewards=rewards,
        pnls=pnls,
        inventories=inventories,
        quoted=quoted,
        actions=actions,
        bid_fills=bid_fills,
        ask_fills=ask_fills,
    )


def run_strategy(
    strategy: BaseStrategy,
    env_config: dict[str, Any],
    *,
    episodes: int,
    seed: int,
) -> AggregateMetrics:
    episode_metrics = [
        run_strategy_episode(strategy, env_config, seed=seed + episode_index)
        for episode_index in range(episodes)
    ]
    return aggregate_episode_metrics(episode_metrics)


def format_summary(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def print_comparison(results: dict[str, AggregateMetrics]) -> None:
    columns = [
        "strategy",
        "total_pnl",
        "total_reward",
        "max_abs_inventory",
        "mean_abs_inventory",
        "final_inventory",
        "number_of_steps",
        "quote_rate",
        "fill_rate",
    ]
    rows = []
    for name, metrics in results.items():
        row = metrics.as_dict()
        rows.append(
            {
                "strategy": name,
                "total_pnl": format_summary(**row["total_pnl"]),
                "total_reward": format_summary(**row["total_reward"]),
                "max_abs_inventory": format_summary(**row["max_abs_inventory"]),
                "mean_abs_inventory": format_summary(**row["mean_abs_inventory"]),
                "final_inventory": format_summary(**row["final_inventory"]),
                "number_of_steps": format_summary(**row["number_of_steps"]),
                "quote_rate": format_summary(**row["quote_rate"]),
                "fill_rate": format_summary(**row["fill_rate"]),
            }
        )

    widths = {
        column: max(len(column), *(len(row[column]) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    separator = "  ".join("-" * widths[column] for column in columns)
    print(header)
    print(separator)
    for row in rows:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mock market-making baselines.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("configs/env_mock.yaml"))
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")

    env_config = load_config(args.config)
    strategies: list[BaseStrategy] = [
        FixedSpreadStrategy(),
        InventorySkewStrategy(),
    ]

    results = {
        strategy.name: run_strategy(
            strategy,
            env_config,
            episodes=args.episodes,
            seed=args.seed,
        )
        for strategy in strategies
    }
    print_comparison(results)


if __name__ == "__main__":
    main()
