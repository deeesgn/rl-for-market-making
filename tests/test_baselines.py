import numpy as np

from rl_mm.backtest.metrics import compute_episode_metrics
from rl_mm.strategies import FixedSpreadStrategy, InventorySkewStrategy


def observation(inventory: float) -> dict[str, np.ndarray]:
    return {"inventory": np.array(inventory, dtype=np.float32)}


def test_fixed_spread_always_selects_medium_quote() -> None:
    strategy = FixedSpreadStrategy()

    assert strategy.select_action(observation(0.0)) == 2
    assert strategy.select_action(observation(10.0)) == 2
    assert strategy.select_action(observation(-10.0)) == 2


def test_inventory_skew_selects_inventory_reducing_actions() -> None:
    strategy = InventorySkewStrategy(threshold=2.0)

    assert strategy.select_action(observation(3.0)) == 4
    assert strategy.select_action(observation(-3.0)) == 5
    assert strategy.select_action(observation(2.0)) == 2
    assert strategy.select_action(observation(-2.0)) == 2
    assert strategy.select_action(observation(0.0)) == 2


def test_compute_episode_metrics() -> None:
    metrics = compute_episode_metrics(
        rewards=[1.0, -0.25, 0.5],
        pnls=[0.5, 0.25, 1.25],
        inventories=[0.0, 2.0, -3.0, -1.0],
    )

    assert metrics.total_pnl == 1.25
    assert metrics.total_reward == 1.25
    assert metrics.max_abs_inventory == 3.0
    assert metrics.final_inventory == -1.0
    assert metrics.number_of_steps == 3
