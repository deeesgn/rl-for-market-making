from pathlib import Path

import yaml

from rl_mm.env import RandomizedMockEnv


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_randomized_env_samples_regime_on_reset(tmp_path: Path) -> None:
    base_config = tmp_path / "base.yaml"
    regime_config = tmp_path / "regime.yaml"
    write_yaml(
        base_config,
        {
            "max_steps": 2,
            "initial_mid_price": 100.0,
            "price_volatility": 0.0,
            "drift": 0.0,
            "fill_probability": 0.0,
            "actions": [1, 2],
        },
    )
    write_yaml(
        regime_config,
        {
            "name": "unit_regime",
            "overrides": {
                "drift": 0.001,
                "price_volatility": 0.002,
            },
        },
    )
    env = RandomizedMockEnv(
        base_config_path=base_config,
        regime_config_paths=[regime_config],
        seed=123,
    )

    observation, info = env.reset()

    assert info["sampled_regime"] == "unit_regime"
    assert env.current_regime_name == "unit_regime"
    assert env.current_config is not None
    assert env.current_config["drift"] == 0.001
    assert env.current_config["price_volatility"] == 0.002
    assert env.observation_space.contains(observation)


def test_randomized_env_step_delegates_to_sampled_mock_env(tmp_path: Path) -> None:
    base_config = tmp_path / "base.yaml"
    regime_config = tmp_path / "regime.yaml"
    write_yaml(
        base_config,
        {
            "max_steps": 1,
            "initial_mid_price": 100.0,
            "price_volatility": 0.0,
            "fill_probability": 0.0,
        },
    )
    write_yaml(regime_config, {"name": "calm", "overrides": {"drift": 0.0}})
    env = RandomizedMockEnv(
        base_config_path=base_config,
        regime_config_paths=[regime_config],
        seed=123,
    )
    env.reset()

    observation, reward, terminated, truncated, info = env.step(2)

    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is True
    assert info["sampled_regime"] == "calm"
    assert info["quoted"] is True
