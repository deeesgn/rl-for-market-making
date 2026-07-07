"""Base interface for rule-based market-making strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class BaseStrategy(ABC):
    """Strategy interface for choosing a discrete mock-env action."""

    name = "base"

    @abstractmethod
    def select_action(self, observation: Mapping[str, Any]) -> int:
        """Choose an action from the current environment observation."""
