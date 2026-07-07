from pathlib import Path

from gymnasium import spaces

from rl_mm.training import load_env_config, make_mock_env, policy_for_env


def test_load_env_config_removes_scripted_actions() -> None:
    config = load_env_config(Path("configs/env_mock.yaml"))

    assert "actions" not in config
    assert config["max_steps"] == 100
    assert config["no_quote_penalty"] == 0.001


def test_policy_for_mock_env_uses_multi_input_policy() -> None:
    env = make_mock_env({"max_steps": 1}, seed=123)

    assert isinstance(env.observation_space, spaces.Dict)
    assert policy_for_env(env) == "MultiInputPolicy"


def test_wrapped_mock_env_observation_values_are_one_dimensional() -> None:
    env = make_mock_env({"max_steps": 1}, seed=123)

    observation, _ = env.reset()

    assert observation["mid_price"].shape == (1,)
    assert observation["inventory"].shape == (1,)
    assert env.observation_space.contains(observation)
