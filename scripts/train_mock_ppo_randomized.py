"""Train PPO on a randomized mix of mock market regimes."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_mm.env.randomized_env import DEFAULT_REGIME_CONFIGS
from rl_mm.training import train_randomized_ppo


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on randomized mock regimes.")
    parser.add_argument("--config", type=Path, default=Path("configs/env_mock.yaml"))
    parser.add_argument("--timesteps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("models/mock_ppo_randomized.zip"))
    parser.add_argument(
        "--regime-config",
        type=Path,
        action="append",
        dest="regime_configs",
        help="Regime config path. Can be supplied multiple times.",
    )
    args = parser.parse_args()

    if args.timesteps < 1:
        raise ValueError("--timesteps must be at least 1")

    saved_path = train_randomized_ppo(
        base_config_path=args.config,
        regime_config_paths=args.regime_configs or DEFAULT_REGIME_CONFIGS,
        timesteps=args.timesteps,
        seed=args.seed,
        model_path=args.model_path,
    )
    print(f"saved randomized PPO model to {saved_path}")


if __name__ == "__main__":
    main()
