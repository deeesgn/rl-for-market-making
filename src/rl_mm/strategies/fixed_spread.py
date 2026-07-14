"""Fixed-spread baseline strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from rl_mm.strategies.base import BaseStrategy


@dataclass(frozen=True)
class FixedSpreadStrategy(BaseStrategy):
    """Place symmetric quotes and widen when configured risk filters trigger."""

    spread_bps: float = 5.0
    volatility_filter: bool = False
    imbalance_filter: bool = False
    volatility_threshold: float = 0.25
    imbalance_threshold: float = 0.5

    name = "fixed_spread"

    def select_action(self, observation: Mapping[str, Any]) -> int:
        volatility = float(observation.get("volatility", 0.0))
        imbalance = float(observation.get("orderbook_imbalance", 0.0))
        if self.volatility_filter and volatility >= self.volatility_threshold:
            return 3
        if self.imbalance_filter and abs(imbalance) >= self.imbalance_threshold:
            return 3
        return 2
