from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rl_mm.strategies import AvellanedaStoikovStrategy
from scripts.run_real_experiment import (
    AvellanedaStoikovParameters,
    MeanStd,
    assert_identical_evaluation_windows,
    select_avellaneda_stoikov_parameters,
)


def test_inventory_shifts_reservation_price_correctly() -> None:
    strategy = AvellanedaStoikovStrategy(gamma=0.01, k=0.5)

    flat = strategy.reservation_price(
        mid_price=100.0,
        inventory=0.0,
        volatility=2.0,
    )
    long = strategy.reservation_price(
        mid_price=100.0,
        inventory=0.01,
        volatility=2.0,
    )
    short = strategy.reservation_price(
        mid_price=100.0,
        inventory=-0.01,
        volatility=2.0,
    )

    assert long < flat < short
    assert np.isclose(flat - long, short - flat)


def test_spread_increases_with_volatility() -> None:
    strategy = AvellanedaStoikovStrategy(gamma=0.01, k=0.5)

    assert strategy.total_spread(3.0) > strategy.total_spread(1.0)


@pytest.mark.parametrize("inventory", [-0.02, -0.01, 0.0, 0.01, 0.02])
def test_quotes_never_cross(inventory: float) -> None:
    strategy = AvellanedaStoikovStrategy(gamma=0.01, k=0.5)
    for mid in (100.5, 102.0, 99.0, 104.0):
        quotes = strategy.quote(
            mid_price=mid,
            best_bid=mid - 0.5,
            best_ask=mid + 0.5,
            inventory=inventory,
        )
        if quotes.bid_price is not None and quotes.ask_price is not None:
            assert quotes.bid_price < quotes.ask_price
        if quotes.bid_price is not None:
            assert quotes.bid_price <= mid - 0.5
        if quotes.ask_price is not None:
            assert quotes.ask_price >= mid + 0.5


def test_quote_prices_are_rounded_to_tick() -> None:
    strategy = AvellanedaStoikovStrategy(gamma=0.005, k=0.5, tick_size=0.1)

    quotes = strategy.quote(
        mid_price=100.05,
        best_bid=100.0,
        best_ask=100.1,
        inventory=0.0,
    )

    assert quotes.bid_price is not None
    assert quotes.ask_price is not None
    assert np.isclose(quotes.bid_price / 0.1, round(quotes.bid_price / 0.1))
    assert np.isclose(quotes.ask_price / 0.1, round(quotes.ask_price / 0.1))


def test_rolling_volatility_uses_only_observed_prefix() -> None:
    strategy = AvellanedaStoikovStrategy(
        gamma=0.01,
        k=0.5,
        volatility_window=60,
    )
    observed = [100.0, 102.0, 101.0, 104.0]
    quotes = None
    for mid in observed:
        quotes = strategy.quote(
            mid_price=mid,
            best_bid=mid - 0.5,
            best_ask=mid + 0.5,
            inventory=0.0,
        )

    assert quotes is not None
    assert np.isclose(quotes.volatility, np.std(np.diff(observed), ddof=0))


def test_validation_selects_gamma_and_k_by_requested_score() -> None:
    first = AvellanedaStoikovParameters(gamma=0.001, k=0.1)
    second = AvellanedaStoikovParameters(gamma=0.005, k=0.5)
    summaries = {
        first: SimpleNamespace(
            total_pnl=MeanStd(1.0, 0.4),
            maximum_drawdown=MeanStd(0.5, 0.0),
        ),
        second: SimpleNamespace(
            total_pnl=MeanStd(1.2, 0.2),
            maximum_drawdown=MeanStd(0.3, 0.0),
        ),
    }

    assert select_avellaneda_stoikov_parameters(summaries) == second


def test_all_strategies_require_identical_test_windows() -> None:
    shared = ("2025-11-07:0:3600:42", "2025-11-08:10:3600:42")
    results = {
        name: SimpleNamespace(window_ids=shared)
        for name in (
            "adaptive_fixed_spread",
            "adaptive_inventory_skew",
            "avellaneda_stoikov",
            "previous_ppo",
            "residual_sac",
        )
    }

    assert assert_identical_evaluation_windows(results) == shared
    results["residual_sac"] = SimpleNamespace(window_ids=("different",))
    with pytest.raises(ValueError, match="identical evaluation windows"):
        assert_identical_evaluation_windows(results)
