import numpy as np

from rl_mm.backtest.metrics import aggregate_episode_metrics, compute_episode_metrics
from rl_mm.strategies import BaseStrategy, FixedSpreadStrategy, InventorySkewStrategy
from scripts.run_baselines import print_comparison, run_strategy


def observation(inventory: float) -> dict[str, np.ndarray]:
    return {"inventory": np.array(inventory, dtype=np.float32)}


class NoQuoteStrategy(BaseStrategy):
    name = "no_quote"

    def select_action(self, observation: dict[str, np.ndarray]) -> int:
        del observation
        return 0


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
        quoted=[True, False, True],
    )

    assert metrics.total_pnl == 1.25
    assert metrics.total_reward == 1.25
    assert metrics.max_abs_inventory == 3.0
    assert metrics.final_inventory == -1.0
    assert metrics.number_of_steps == 3
    assert metrics.quoted_steps == 2
    assert metrics.quote_rate == 2 / 3


def test_aggregate_episode_metrics() -> None:
    first = compute_episode_metrics(
        rewards=[1.0, 2.0],
        pnls=[0.5, 1.0],
        inventories=[0.0, 1.0],
        quoted=[True, True],
    )
    second = compute_episode_metrics(
        rewards=[-1.0, 1.0],
        pnls=[-0.5, 0.0],
        inventories=[0.0, -3.0],
        quoted=[False, True],
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
    assert aggregate.quoted_steps.mean == 1.5
    assert aggregate.quoted_steps.std == 0.5
    assert aggregate.quote_rate.mean == 0.75
    assert aggregate.quote_rate.std == 0.25


def test_run_strategy_aggregates_multiple_seeded_episodes() -> None:
    aggregate = run_strategy(
        FixedSpreadStrategy(),
        {"max_steps": 3, "initial_mid_price": 100.0},
        episodes=4,
        seed=10,
    )

    assert aggregate.number_of_steps.mean == 3.0
    assert aggregate.number_of_steps.std == 0.0
    assert aggregate.max_abs_inventory.mean >= 0.0
    assert aggregate.quote_rate.mean == 1.0


def test_run_strategy_is_deterministic_for_same_seed() -> None:
    env_config = {"max_steps": 3, "initial_mid_price": 100.0}

    first = run_strategy(InventorySkewStrategy(), env_config, episodes=4, seed=10)
    second = run_strategy(InventorySkewStrategy(), env_config, episodes=4, seed=10)

    assert first.as_dict() == second.as_dict()


def test_print_comparison_reports_mean_plus_std(capsys) -> None:
    aggregate = run_strategy(
        FixedSpreadStrategy(),
        {"max_steps": 2, "initial_mid_price": 100.0},
        episodes=2,
        seed=20,
    )

    print_comparison({"fixed_spread": aggregate})

    output = capsys.readouterr().out
    assert "strategy" in output
    assert "total_pnl" in output
    assert "total_reward" in output
    assert "max_abs_inventory" in output
    assert "final_inventory" in output
    assert "number_of_steps" in output
    assert "quote_rate" in output
    assert "fixed_spread" in output
    assert "+/-" in output


def test_quote_rate_is_zero_for_always_no_quote() -> None:
    aggregate = run_strategy(NoQuoteStrategy(), {"max_steps": 3}, episodes=2, seed=1)

    assert aggregate.quote_rate.mean == 0.0
    assert aggregate.quoted_steps.mean == 0.0


def test_quote_rate_is_positive_for_fixed_spread() -> None:
    aggregate = run_strategy(FixedSpreadStrategy(), {"max_steps": 3}, episodes=2, seed=1)

    assert aggregate.quote_rate.mean > 0.0
    assert aggregate.quoted_steps.mean > 0.0
