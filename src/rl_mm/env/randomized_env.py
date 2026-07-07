"""Randomized mock market environment that samples regimes on reset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml

from rl_mm.env.mock_market_env import MockMarketMakingEnv

DEFAULT_REGIME_CONFIGS = [
    Path("configs/regimes/calm.yaml"),
    Path("configs/regimes/high_volatility.yaml"),
    Path("configs/regimes/trend_up.yaml"),
    Path("configs/regimes/trend_down.yaml"),
    Path("configs/regimes/toxic_flow.yaml"),
]


class RandomizedMockEnv(gym.Env):
    """Sample a mock market regime on every reset."""

    metadata = MockMarketMakingEnv.metadata

    def __init__(
        self,
        *,
        base_config_path: Path = Path("configs/env_mock.yaml"),
        regime_config_paths: list[Path] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.base_config_path = base_config_path
        self.regime_config_paths = regime_config_paths or DEFAULT_REGIME_CONFIGS
        self.initial_seed = seed
        self.base_config = load_env_config(base_config_path)
        self.regimes = [load_regime_config(path) for path in self.regime_config_paths]
        if not self.regimes:
            raise ValueError("At least one regime config is required.")

        template_env = MockMarketMakingEnv(**self.base_config)
        self.action_space = template_env.action_space
        self.observation_space = template_env.observation_space
        template_env.close()

        self.current_env: MockMarketMakingEnv | None = None
        self.current_regime_name: str | None = None
        self.current_config: dict[str, Any] | None = None
        self._has_seeded = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options
        if seed is not None:
            super().reset(seed=seed)
            self._has_seeded = True
        elif not self._has_seeded:
            super().reset(seed=self.initial_seed)
            self._has_seeded = True

        regime_index = int(self.np_random.integers(len(self.regimes)))
        regime_name, overrides = self.regimes[regime_index]
        env_seed = int(self.np_random.integers(0, np.iinfo(np.int32).max))
        env_config = {**self.base_config, **overrides, "seed": env_seed}

        self.current_env = MockMarketMakingEnv(**env_config)
        self.current_regime_name = regime_name
        self.current_config = env_config
        observation, info = self.current_env.reset()
        info = {**info, "sampled_regime": regime_name}
        return observation, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.current_env is None:
            raise RuntimeError("RandomizedMockEnv must be reset before calling step().")

        observation, reward, terminated, truncated, info = self.current_env.step(action)
        info = {**info, "sampled_regime": self.current_regime_name}
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        if self.current_env is None:
            return
        self.current_env.render()

    def close(self) -> None:
        if self.current_env is not None:
            self.current_env.close()


def load_env_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    config.pop("actions", None)
    return config


def load_regime_config(path: Path) -> tuple[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")

    name = str(config.get("name", path.stem))
    overrides = config.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"Expected 'overrides' mapping in {path}")
    return name, overrides
