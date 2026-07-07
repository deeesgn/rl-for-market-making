"""Fixed-spread baseline strategy."""

from __future__ import annotations

from typing import Any, Mapping

from rl_mm.strategies.base import BaseStrategy


class FixedSpreadStrategy(BaseStrategy):
    """Always place the medium symmetric quote."""

    name = "fixed_spread"

    def select_action(self, observation: Mapping[str, Any]) -> int:
        del observation
        return 2
