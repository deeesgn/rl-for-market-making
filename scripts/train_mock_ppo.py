"""Train a smoke-level PPO agent on the mock market-making environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_mm.training import train_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on MockMarketMakingEnv.")
    parser.add_argument("--config", type=Path, default=Path("configs/env_mock.yaml"))
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("models/mock_ppo.zip"))
    args = parser.parse_args()

    if args.timesteps < 1:
        raise ValueError("--timesteps must be at least 1")

    saved_path = train_ppo(
        config_path=args.config,
        timesteps=args.timesteps,
        seed=args.seed,
        model_path=args.model_path,
    )
    print(f"saved PPO model to {saved_path}")


if __name__ == "__main__":
    main()
