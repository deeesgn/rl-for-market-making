"""PPO training and evaluation utilities for the mock market-making env."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from stable_baselines3 import PPO

from rl_mm.backtest.metrics import (
    AggregateMetrics,
    EpisodeMetrics,
    aggregate_episode_metrics,
    compute_episode_metrics,
)
from rl_mm.env import MockMarketMakingEnv


class ScalarDictObservationWrapper(gym.ObservationWrapper):
    """Convert scalar dict observations to one-dimensional arrays for SB3."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            return

        self.observation_space = spaces.Dict(
            {
                key: spaces.Box(
                    low=np.asarray(space.low, dtype=space.dtype).reshape(1),
                    high=np.asarray(space.high, dtype=space.dtype).reshape(1),
                    shape=(1,),
                    dtype=space.dtype,
                )
                for key, space in env.observation_space.spaces.items()
            }
        )

    def observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            key: np.asarray(value, dtype=self.observation_space[key].dtype).reshape(1)
            for key, value in observation.items()
        }


def load_env_config(path: Path) -> dict[str, Any]:
    """Load mock environment config and ignore scripted action sequences."""

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    config.pop("actions", None)
    return config


def make_mock_env(env_config: dict[str, Any], *, seed: int) -> gym.Env:
    """Create the mock env used by PPO training and evaluation."""

    env = MockMarketMakingEnv(**{**env_config, "seed": seed})
    return ScalarDictObservationWrapper(env)


def policy_for_env(env: gym.Env) -> str:
    """Pick the SB3 policy class name from the environment observation space."""

    if isinstance(env.observation_space, spaces.Dict):
        return "MultiInputPolicy"
    return "MlpPolicy"


def train_ppo(
    *,
    config_path: Path,
    timesteps: int,
    seed: int,
    model_path: Path,
) -> Path:
    """Train a small PPO model on the mock environment and save it."""

    env_config = load_env_config(config_path)
    env = make_mock_env(env_config, seed=seed)
    model = PPO(
        policy_for_env(env),
        env,
        seed=seed,
        verbose=0,
        n_steps=64,
        batch_size=64,
        n_epochs=2,
    )
    model.learn(total_timesteps=timesteps)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    env.close()
    return model_path


def evaluate_ppo(
    *,
    config_path: Path,
    model_path: Path,
    episodes: int,
    seed: int,
) -> AggregateMetrics:
    """Evaluate a saved PPO model over multiple seeded mock episodes."""

    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    env_config = load_env_config(config_path)
    model = PPO.load(model_path)
    episode_metrics = [
        run_model_episode(model, env_config=env_config, seed=seed + episode_index)
        for episode_index in range(episodes)
    ]
    return aggregate_episode_metrics(episode_metrics)


def run_model_episode(PPO_model: PPO, *, env_config: dict[str, Any], seed: int) -> EpisodeMetrics:
    """Run one deterministic PPO evaluation episode."""

    env = make_mock_env(env_config, seed=seed)
    observation, _ = env.reset()
    rewards: list[float] = []
    pnls: list[float] = []
    inventories = [float(observation["inventory"][0])]
    quoted: list[bool] = []

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = PPO_model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(int(action))
        rewards.append(float(reward))
        pnls.append(float(info["pnl"]))
        inventories.append(float(observation["inventory"][0]))
        quoted.append(bool(info["quoted"]))

    env.close()
    return compute_episode_metrics(
        rewards=rewards,
        pnls=pnls,
        inventories=inventories,
        quoted=quoted,
    )
