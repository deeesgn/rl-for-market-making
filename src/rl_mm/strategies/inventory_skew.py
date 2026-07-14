"""Inventory-aware skew baseline strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from rl_mm.strategies.base import BaseStrategy


@dataclass(frozen=True)
class InventorySkewStrategy(BaseStrategy):
    """Use adaptive spreads and skew toward reducing material inventory."""

    threshold: float = 1.0
    spread_bps: float = 5.0
    volatility_filter: bool = False
    imbalance_filter: bool = False
    volatility_threshold: float = 0.25
    imbalance_threshold: float = 0.5

    name = "inventory_skew"

    def select_action(self, observation: Mapping[str, Any]) -> int:
        inventory_value = (
            observation["inventory_ratio"]
            if "inventory_ratio" in observation
            else observation["inventory"]
        )
        inventory = float(inventory_value)
        if inventory > self.threshold:
            return 4
        if inventory < -self.threshold:
            return 5
        volatility = float(observation.get("volatility", 0.0))
        imbalance = float(observation.get("orderbook_imbalance", 0.0))
        if self.volatility_filter and volatility >= self.volatility_threshold:
            return 3
        if self.imbalance_filter and abs(imbalance) >= self.imbalance_threshold:
            return 3
        return 2
