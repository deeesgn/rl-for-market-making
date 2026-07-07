"""Evaluate a randomized-regime PPO model across all fixed mock regimes."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_mm.env.randomized_env import DEFAULT_REGIME_CONFIGS

try:
    from scripts.run_baselines import print_comparison
    from scripts.run_mock_regime_stress import run_stress
except ModuleNotFoundError:
    from run_baselines import print_comparison
    from run_mock_regime_stress import run_stress


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate randomized PPO across regimes.")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base-config", type=Path, default=Path("configs/env_mock.yaml"))
    parser.add_argument("--model-path", type=Path, default=Path("models/mock_ppo_randomized.zip"))
    parser.add_argument(
        "--regime-config",
        type=Path,
        action="append",
        dest="regime_configs",
        help="Regime config path. Can be supplied multiple times.",
    )
    args = parser.parse_args()

    results = run_stress(
        base_config_path=args.base_config,
        regime_paths=args.regime_configs or DEFAULT_REGIME_CONFIGS,
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
