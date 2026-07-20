"""Avellaneda-Stoikov quotes for the real orderbook replay."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class AvellanedaStoikovQuotes:
    """Passive quote intent produced from information available at the current step."""

    bid_price: float | None
    ask_price: float | None
    bid_size: float
    ask_size: float
    reservation_price: float
    total_spread: float
    volatility: float


@dataclass
class AvellanedaStoikovStrategy:
    """Stateful AS baseline using a trailing window of one-second mid-price changes."""

    gamma: float
    k: float
    tick_size: float = 0.1
    max_inventory_btc: float = 0.02
    order_size_btc: float = 0.001
    volatility_window: int = 60
    horizon_seconds: float = 60.0
    _mid_prices: deque[float] = field(init=False, repr=False)

    name = "avellaneda_stoikov"

    def __post_init__(self) -> None:
        if self.gamma <= 0.0 or self.k <= 0.0:
            raise ValueError("gamma and k must be positive")
        if self.tick_size <= 0.0:
            raise ValueError("tick_size must be positive")
        if self.max_inventory_btc <= 0.0 or self.order_size_btc <= 0.0:
            raise ValueError("inventory and order sizes must be positive")
        if self.volatility_window < 2 or self.horizon_seconds <= 0.0:
            raise ValueError("volatility window and horizon must be positive")
        self._mid_prices = deque(maxlen=self.volatility_window + 1)

    def reservation_price(
        self,
        *,
        mid_price: float,
        inventory: float,
        volatility: float,
    ) -> float:
        return (
            mid_price
            - inventory
            * self.gamma
            * volatility**2
            * self.horizon_seconds
        )

    def total_spread(self, volatility: float) -> float:
        return (
            self.gamma * volatility**2 * self.horizon_seconds
            - (2.0 / self.gamma) * math.log1p(self.gamma / self.k)
        )

    def update_volatility(self, mid_price: float) -> float:
        """Append only the current mid and calculate trailing absolute-price volatility."""
        self._mid_prices.append(float(mid_price))
        if len(self._mid_prices) < 3:
            return 0.0
        changes = np.diff(np.asarray(self._mid_prices, dtype=float))
        return float(np.std(changes, ddof=0))

    def quote(
        self,
        *,
        mid_price: float,
        best_bid: float,
        best_ask: float,
        inventory: float,
    ) -> AvellanedaStoikovQuotes:
        volatility = self.update_volatility(mid_price)
        reservation = self.reservation_price(
            mid_price=mid_price,
            inventory=inventory,
            volatility=volatility,
        )
        raw_spread = self.total_spread(volatility)
        # The requested formula can be negative at low volatility. A two-tick floor
        # is the minimum admissible spread and prevents crossed quotes.
        admissible_spread = max(raw_spread, 2.0 * self.tick_size)
        bid_target = min(best_bid, reservation - admissible_spread / 2.0)
        ask_target = max(best_ask, reservation + admissible_spread / 2.0)
        bid_price = self._round_down(bid_target)
        ask_price = self._round_up(ask_target)
        if bid_price >= ask_price:
            bid_price = self._round_down(ask_price - self.tick_size)

        buy_capacity = max(0.0, self.max_inventory_btc - inventory)
        sell_capacity = max(0.0, self.max_inventory_btc + inventory)
        bid_size = min(self.order_size_btc, buy_capacity)
        ask_size = min(self.order_size_btc, sell_capacity)
        if bid_size <= 1e-12:
            bid_price = None
            bid_size = 0.0
        if ask_size <= 1e-12:
            ask_price = None
            ask_size = 0.0
        return AvellanedaStoikovQuotes(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            reservation_price=reservation,
            total_spread=raw_spread,
            volatility=volatility,
        )

    def _round_down(self, price: float) -> float:
        ticks = math.floor((price + self.tick_size * 1e-9) / self.tick_size)
        return round(ticks * self.tick_size, 10)

    def _round_up(self, price: float) -> float:
        ticks = math.ceil((price - self.tick_size * 1e-9) / self.tick_size)
        return round(ticks * self.tick_size, 10)
