"""Minimal Gymnasium environment for mock market-making mechanics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class QuoteSpec:
    """Distances from mid price for bid and ask quotes."""

    bid_distance: float
    ask_distance: float
    name: str


class MockMarketMakingEnv(gym.Env):
    """Simple seeded market-making environment with synthetic fills.

    The environment is intentionally small and mechanical. It is useful for
    checking accounting, inventory risk, and Gymnasium integration before any
    historical data or backtesting engine is introduced.
    """

    metadata = {"render_modes": ["human"]}

    ACTIONS: dict[int, QuoteSpec | None] = {
        0: None,
        1: QuoteSpec(bid_distance=0.01, ask_distance=0.01, name="narrow_symmetric"),
        2: QuoteSpec(bid_distance=0.03, ask_distance=0.03, name="medium_symmetric"),
        3: QuoteSpec(bid_distance=0.05, ask_distance=0.05, name="wide_symmetric"),
        4: QuoteSpec(bid_distance=0.05, ask_distance=0.01, name="skew_to_sell_inventory"),
        5: QuoteSpec(bid_distance=0.01, ask_distance=0.05, name="skew_to_buy_inventory"),
    }

    def __init__(
        self,
        *,
        seed: int | None = None,
        max_steps: int = 100,
        initial_mid_price: float = 100.0,
        initial_cash: float = 0.0,
        initial_inventory: int = 0,
        order_size: int = 1,
        price_volatility: float = 0.001,
        drift: float = 0.0,
        inventory_penalty: float = 0.01,
        no_quote_penalty: float = 0.001,
        fill_probability: float = 0.45,
        bid_fill_multiplier: float = 1.0,
        ask_fill_multiplier: float = 1.0,
        fill_decay: float = 0.03,
        return_window: int = 20,
    ) -> None:
        super().__init__()
        self.initial_seed = seed
        self.max_steps = max_steps
        self.initial_mid_price = initial_mid_price
        self.initial_cash = initial_cash
        self.initial_inventory = initial_inventory
        self.order_size = order_size
        self.price_volatility = price_volatility
        self.drift = drift
        self.inventory_penalty = inventory_penalty
        self.no_quote_penalty = no_quote_penalty
        self.fill_probability = fill_probability
        self.bid_fill_multiplier = bid_fill_multiplier
        self.ask_fill_multiplier = ask_fill_multiplier
        self.fill_decay = fill_decay
        self.return_window = return_window

        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Dict(
            {
                "mid_price": spaces.Box(0.0, np.inf, shape=(), dtype=np.float32),
                "inventory": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
                "cash": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
                "spread": spaces.Box(0.0, np.inf, shape=(), dtype=np.float32),
                "recent_return": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
                "volatility": spaces.Box(0.0, np.inf, shape=(), dtype=np.float32),
            }
        )

        self.step_count = 0
        self.mid_price = self.initial_mid_price
        self.inventory = self.initial_inventory
        self.cash = self.initial_cash
        self.spread = 0.0
        self.recent_return = 0.0
        self.volatility = 0.0
        self.portfolio_value = self.initial_cash + self.initial_inventory * self.initial_mid_price
        self.pnl = 0.0
        self._returns: list[float] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options
        super().reset(seed=self.initial_seed if seed is None else seed)

        self.step_count = 0
        self.mid_price = self.initial_mid_price
        self.inventory = self.initial_inventory
        self.cash = self.initial_cash
        self.spread = 0.0
        self.recent_return = 0.0
        self.volatility = 0.0
        self.portfolio_value = self.cash + self.inventory * self.mid_price
        self.pnl = 0.0
        self._returns = []

        info = self._info(action=0, bid_filled=False, ask_filled=False)
        return self._observation(), info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected an integer in [0, 5].")

        previous_value = self.portfolio_value
        quote = self.ACTIONS[int(action)]
        bid_filled = False
        ask_filled = False
        bid_price = None
        ask_price = None

        if quote is None:
            self.spread = 0.0
        else:
            bid_price = self.mid_price - quote.bid_distance
            ask_price = self.mid_price + quote.ask_distance
            self.spread = quote.bid_distance + quote.ask_distance

            bid_filled = self._is_filled(quote.bid_distance, self.bid_fill_multiplier)
            ask_filled = self._is_filled(quote.ask_distance, self.ask_fill_multiplier)

            if bid_filled:
                self.inventory += self.order_size
                self.cash -= bid_price * self.order_size

            if ask_filled:
                self.inventory -= self.order_size
                self.cash += ask_price * self.order_size

        self._simulate_mid_price()
        self.step_count += 1

        self.portfolio_value = self.cash + self.inventory * self.mid_price
        initial_value = self.initial_cash + self.initial_inventory * self.initial_mid_price
        self.pnl = self.portfolio_value - initial_value
        value_change = self.portfolio_value - previous_value
        inventory_cost = self.inventory_penalty * float(self.inventory**2)
        no_quote_cost = self.no_quote_penalty if int(action) == 0 else 0.0
        reward = value_change - inventory_cost - no_quote_cost

        terminated = False
        truncated = self.step_count >= self.max_steps
        info = self._info(
            action=int(action),
            bid_filled=bid_filled,
            ask_filled=ask_filled,
            bid_price=bid_price,
            ask_price=ask_price,
            reward=reward,
            inventory_penalty=inventory_cost,
            no_quote_penalty=no_quote_cost,
        )
        return self._observation(), float(reward), terminated, truncated, info

    def render(self) -> None:
        print(
            f"step={self.step_count} mid={self.mid_price:.4f} "
            f"inventory={self.inventory} cash={self.cash:.4f} "
            f"portfolio={self.portfolio_value:.4f} pnl={self.pnl:.4f}"
        )

    def _is_filled(self, distance: float, multiplier: float) -> bool:
        probability = self.fill_probability * multiplier * np.exp(-distance / self.fill_decay)
        probability = float(np.clip(probability, 0.0, 1.0))
        return bool(self.np_random.random() < probability)

    def _simulate_mid_price(self) -> None:
        previous_mid = self.mid_price
        price_change = self.np_random.normal(self.drift, self.price_volatility)
        self.mid_price = max(0.01, self.mid_price * (1.0 + price_change))
        self.recent_return = self.mid_price / previous_mid - 1.0
        self._returns.append(self.recent_return)
        self._returns = self._returns[-self.return_window :]
        self.volatility = float(np.std(self._returns)) if len(self._returns) > 1 else 0.0

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "mid_price": np.array(self.mid_price, dtype=np.float32),
            "inventory": np.array(self.inventory, dtype=np.float32),
            "cash": np.array(self.cash, dtype=np.float32),
            "spread": np.array(self.spread, dtype=np.float32),
            "recent_return": np.array(self.recent_return, dtype=np.float32),
            "volatility": np.array(self.volatility, dtype=np.float32),
        }

    def _info(
        self,
        *,
        action: int,
        bid_filled: bool,
        ask_filled: bool,
        bid_price: float | None = None,
        ask_price: float | None = None,
        reward: float = 0.0,
        inventory_penalty: float = 0.0,
        no_quote_penalty: float = 0.0,
    ) -> dict[str, Any]:
        quote = self.ACTIONS[action]
        action_name = "no_quote" if quote is None else quote.name
        quoted = quote is not None
        return {
            "step": self.step_count,
            "action_name": action_name,
            "quoted": quoted,
            "bid_filled": bid_filled,
            "ask_filled": ask_filled,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "portfolio_value": self.portfolio_value,
            "pnl": self.pnl,
            "reward": reward,
            "inventory_penalty": inventory_penalty,
            "no_quote_penalty": no_quote_penalty,
        }
