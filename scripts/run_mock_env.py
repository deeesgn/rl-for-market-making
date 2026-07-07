"""Run a short episode in the mock market-making environment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from rl_mm.env import MockMarketMakingEnv


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mock market-making environment.")
    parser.add_argument("--config", type=Path, default=Path("configs/env_mock.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    actions = list(config.pop("actions", []))
    env = MockMarketMakingEnv(**config)
    observation, info = env.reset()

    print("mock environment reset")
    print(
        f"initial mid={float(observation['mid_price']):.4f} "
        f"inventory={float(observation['inventory']):.0f} "
        f"cash={float(observation['cash']):.4f}"
    )

    total_reward = 0.0
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        print(
            f"step={info['step']:02d} action={info['action_name']} "
            f"bid_fill={info['bid_filled']} ask_fill={info['ask_filled']} "
            f"mid={float(observation['mid_price']):.4f} "
            f"inventory={float(observation['inventory']):.0f} "
            f"cash={float(observation['cash']):.4f} "
            f"pnl={info['pnl']:.4f} reward={reward:.4f}"
        )
        if terminated or truncated:
            break

    print(f"done: portfolio_value={info['portfolio_value']:.4f} total_reward={total_reward:.4f}")


if __name__ == "__main__":
    main()
