import numpy as np

from rl_mm.backtest.metrics import aggregate_episode_metrics, compute_episode_metrics
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


def test_aggregate_episode_metrics() -> None:
    first = compute_episode_metrics(
        rewards=[1.0, 2.0],
        pnls=[0.5, 1.0],
        inventories=[0.0, 1.0],
    )
    second = compute_episode_metrics(
        rewards=[-1.0, 1.0],
        pnls=[-0.5, 0.0],
        inventories=[0.0, -3.0],
    )

    aggregate = aggregate_episode_metrics([first, second])

    assert aggregate.total_pnl.mean == 0.5
    assert aggregate.total_pnl.std == 0.5
    assert aggregate.total_reward.mean == 1.5
    assert aggregate.total_reward.std == 1.5
    assert aggregate.max_abs_inventory.mean == 2.0
    assert aggregate.max_abs_inventory.std == 1.0
    assert aggregate.final_inventory.mean == -1.0
    assert aggregate.final_inventory.std == 2.0
    assert aggregate.number_of_steps.mean == 2.0
    assert aggregate.number_of_steps.std == 0.0
