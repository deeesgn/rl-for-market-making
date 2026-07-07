"""Evaluate a trained PPO agent on the mock market-making environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_mm.training import evaluate_ppo


def format_summary(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PPO on MockMarketMakingEnv.")
    parser.add_argument("--config", type=Path, default=Path("configs/env_mock.yaml"))
    parser.add_argument("--model-path", type=Path, default=Path("models/mock_ppo.zip"))
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    metrics = evaluate_ppo(
        config_path=args.config,
        model_path=args.model_path,
        episodes=args.episodes,
        seed=args.seed,
    )
    row = metrics.as_dict()

    print("metric             mean +/- std")
    print("-----------------  -----------------")
    for name in [
        "total_pnl",
        "total_reward",
        "max_abs_inventory",
        "final_inventory",
        "number_of_steps",
        "quote_rate",
    ]:
        print(f"{name.ljust(17)}  {format_summary(**row[name])}")


if __name__ == "__main__":
    main()
