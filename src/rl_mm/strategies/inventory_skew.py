"""Inventory-aware skew baseline strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from rl_mm.strategies.base import BaseStrategy


@dataclass(frozen=True)
class InventorySkewStrategy(BaseStrategy):
    """Skew quotes when inventory drifts beyond a threshold."""

    threshold: float = 1.0

    name = "inventory_skew"

    def select_action(self, observation: Mapping[str, Any]) -> int:
        inventory = float(observation["inventory"])
        if inventory > self.threshold:
            return 4
        if inventory < -self.threshold:
            return 5
        return 2
