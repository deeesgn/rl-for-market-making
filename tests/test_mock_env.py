import numpy as np

from rl_mm.env import MockMarketMakingEnv

EXPECTED_KEYS = {
    "mid_price",
    "inventory",
    "cash",
    "spread",
    "recent_return",
    "volatility",
}


def test_reset_returns_expected_observation() -> None:
    env = MockMarketMakingEnv(seed=123)

    observation, info = env.reset()

    assert set(observation) == EXPECTED_KEYS
    assert env.observation_space.contains(observation)
    assert float(observation["mid_price"]) == 100.0
    assert float(observation["inventory"]) == 0.0
    assert info["portfolio_value"] == 0.0


def test_step_updates_accounting_and_reward() -> None:
    env = MockMarketMakingEnv(seed=123, max_steps=1)
    env.reset()

    observation, reward, terminated, truncated, info = env.step(1)

    assert set(observation) == EXPECTED_KEYS
    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is True
    assert "portfolio_value" in info
    assert "pnl" in info
    assert info["action_name"] == "narrow_symmetric"


def test_same_seed_and_actions_are_deterministic() -> None:
    actions = [1, 2, 4, 5, 3, 0]
    first = run_episode(actions)
    second = run_episode(actions)

    np.testing.assert_allclose(first, second)


def run_episode(actions: list[int]) -> np.ndarray:
    env = MockMarketMakingEnv(seed=7)
    observation, _ = env.reset()
    rewards = []
    for action in actions:
        observation, reward, _, _, info = env.step(action)
        rewards.append(reward)

    return np.array(
        [
            float(observation["mid_price"]),
            float(observation["inventory"]),
            float(observation["cash"]),
            info["portfolio_value"],
            info["pnl"],
            *rewards,
        ]
    )
