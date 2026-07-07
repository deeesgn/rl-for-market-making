"""Run rule-based baselines in the mock market-making environment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from rl_mm.backtest.metrics import EpisodeMetrics, compute_episode_metrics
from rl_mm.env import MockMarketMakingEnv
from rl_mm.strategies import BaseStrategy, FixedSpreadStrategy, InventorySkewStrategy


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    config.pop("actions", None)
    return config


def run_strategy(strategy: BaseStrategy, env_config: dict[str, Any]) -> EpisodeMetrics:
    env = MockMarketMakingEnv(**env_config)
    observation, _ = env.reset()

    rewards: list[float] = []
    pnls: list[float] = []
    inventories = [float(observation["inventory"])]

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = strategy.select_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        pnls.append(float(info["pnl"]))
        inventories.append(float(observation["inventory"]))

    return compute_episode_metrics(rewards=rewards, pnls=pnls, inventories=inventories)


def print_comparison(results: dict[str, EpisodeMetrics]) -> None:
    columns = [
        "strategy",
        "total_pnl",
        "total_reward",
        "max_abs_inventory",
        "final_inventory",
        "number_of_steps",
    ]
    rows = []
    for name, metrics in results.items():
        row = metrics.as_dict()
        rows.append(
            {
                "strategy": name,
                "total_pnl": f"{row['total_pnl']:.4f}",
                "total_reward": f"{row['total_reward']:.4f}",
                "max_abs_inventory": f"{row['max_abs_inventory']:.0f}",
                "final_inventory": f"{row['final_inventory']:.0f}",
                "number_of_steps": str(row["number_of_steps"]),
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
    parser.add_argument("--config", type=Path, default=Path("configs/env_mock.yaml"))
    args = parser.parse_args()

    env_config = load_config(args.config)
    strategies: list[BaseStrategy] = [
        FixedSpreadStrategy(),
        InventorySkewStrategy(),
    ]

    results = {strategy.name: run_strategy(strategy, env_config) for strategy in strategies}
    print_comparison(results)


if __name__ == "__main__":
    main()
