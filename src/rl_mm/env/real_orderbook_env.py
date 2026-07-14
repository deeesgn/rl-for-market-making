"""Gymnasium replay environment for processed Bybit orderbooks and trades."""

from __future__ import annotations

from collections import deque
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from rl_mm.strategies import InventorySkewStrategy


class RealOrderbookEnv(gym.Env):
    """Replay one daily Top-10 parquet file with trade- and queue-driven fills."""

    QUOTE_WIDTH_MULTIPLIERS = {
        1: (0.5, 0.5),
        2: (1.0, 1.0),
        3: (2.0, 2.0),
        4: (1.5, 0.25),
        5: (0.25, 1.5),
    }
    SIZE_MULTIPLIERS = np.array([0.5, 1.0, 2.0], dtype=float)
    BOOK_COLUMNS = [
        column
        for level in range(1, 11)
        for column in (
            f"bid_price_{level}",
            f"bid_size_{level}",
            f"ask_price_{level}",
            f"ask_size_{level}",
        )
    ]
    MARKET_COLUMNS = ["mid_price", "spread", "orderbook_imbalance"]
    REQUIRED_COLUMNS = ["timestamp", *BOOK_COLUMNS, *MARKET_COLUMNS]
    INVENTORY_PENALTY = 0.01
    INVENTORY_QUOTE_CUTOFF = 0.8
    OBSERVATION_SPREAD_SCALE_BPS = 20.0
    OBSERVATION_RETURN_SCALE = 1_000.0
    RESIDUAL_FEATURES = 11
    RESIDUAL_VARIANTS = {"legacy", "conservative", "selective_narrow", "asymmetric"}

    def __init__(
        self,
        *,
        data_dir: str | Path,
        trades_dir: str | Path | None = None,
        start_date: str,
        end_date: str,
        symbol: str = "BTCUSDT",
        initial_cash: float = 10_000.0,
        max_inventory_btc: float = 0.02,
        order_size_btc: float = 0.001,
        maker_fee: float = 0.0002,
        episode_steps: int | None = None,
        seed: int | None = None,
        random_start: bool = False,
        mandatory_quoting: bool = False,
        queue_fraction: float = 0.5,
        quote_spread_bps: float = 5.0,
        inventory_penalty_multiplier: float = 1.0,
        size_multipliers: tuple[float, float, float] = (0.5, 1.0, 2.0),
        quote_offset_scale: float = 1.0,
        residual_continuous: bool = False,
        residual_variant: str = "legacy",
        sequence_length: int = 30,
        base_spread_bps: float = 5.0,
        base_volatility_filter: bool = False,
        base_imbalance_filter: bool = False,
    ) -> None:
        super().__init__()
        if max_inventory_btc <= 0:
            raise ValueError("max_inventory_btc must be positive")
        if order_size_btc <= 0 or order_size_btc > max_inventory_btc:
            raise ValueError("order_size_btc must be positive and no larger than max inventory")
        if maker_fee < 0:
            raise ValueError("maker_fee must be non-negative")
        if episode_steps is not None and episode_steps < 1:
            raise ValueError("episode_steps must be positive")
        if not 0.0 <= queue_fraction <= 1.0:
            raise ValueError("queue_fraction must be between 0 and 1")
        if quote_spread_bps <= 0:
            raise ValueError("quote_spread_bps must be positive")
        if inventory_penalty_multiplier < 0:
            raise ValueError("inventory_penalty_multiplier must be non-negative")
        if len(size_multipliers) != 3 or any(value <= 0 for value in size_multipliers):
            raise ValueError("size_multipliers must contain three positive values")
        if quote_offset_scale <= 0:
            raise ValueError("quote_offset_scale must be positive")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if residual_variant not in self.RESIDUAL_VARIANTS:
            raise ValueError(f"Unknown residual variant: {residual_variant}")

        self.data_dir = Path(data_dir)
        self.trades_dir = Path(trades_dir) if trades_dir is not None else None
        self.symbol = symbol.upper()
        self.start_date = date.fromisoformat(start_date)
        self.end_date = date.fromisoformat(end_date)
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        self.initial_cash = float(initial_cash)
        self.max_inventory_btc = float(max_inventory_btc)
        self.order_size_btc = float(order_size_btc)
        self.maker_fee = float(maker_fee)
        self.episode_steps = episode_steps
        self.initial_seed = seed
        self.random_start = random_start
        self.mandatory_quoting = bool(mandatory_quoting)
        self.queue_fraction = float(queue_fraction)
        self.quote_spread_bps = float(quote_spread_bps)
        self.inventory_penalty_multiplier = float(inventory_penalty_multiplier)
        self.size_multipliers = np.asarray(size_multipliers, dtype=float)
        self.quote_offset_scale = float(quote_offset_scale)
        self.residual_continuous = bool(residual_continuous)
        self.residual_variant = residual_variant
        self.sequence_length = int(sequence_length)
        self.base_strategy = InventorySkewStrategy(
            threshold=0.25,
            spread_bps=base_spread_bps,
            volatility_filter=base_volatility_filter,
            imbalance_filter=base_imbalance_filter,
        )
        self._has_reset = False
        self._daily_files = self._find_daily_files()
        if not self._daily_files:
            raise FileNotFoundError(
                f"No {self.symbol} orderbook parquet files found in {self.data_dir} "
                f"from {start_date} to {end_date}"
            )

        if self.residual_continuous:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
            self.observation_space = spaces.Box(
                -1.0,
                1.0,
                shape=(self.sequence_length, self.RESIDUAL_FEATURES),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.MultiDiscrete(
                [5 if self.mandatory_quoting else 6, 3]
            )
            self.observation_space = spaces.Box(
                low=np.array(
                    [-1.0, 0.0, -1.0, -1.0, 0.0, -1.0, 0.0, 0.0],
                    dtype=np.float32,
                ),
                high=np.ones(8, dtype=np.float32),
                dtype=np.float32,
            )

        self.cash = self.initial_cash
        self.inventory = 0.0
        self.equity = self.initial_cash
        self.current_date: date | None = None
        self.current_index = 0
        self.step_count = 0
        self.max_steps = 0
        self.total_fees = 0.0
        self.total_turnover = 0.0
        self.bid_fill_count = 0
        self.ask_fill_count = 0
        self.total_fill_count = 0
        self.inventory_penalty_total = 0.0
        self._data: dict[str, np.ndarray] = {}
        self._trade_data: dict[str, np.ndarray] = {}
        self._mid_returns = np.array([], dtype=float)
        self._short_volatility = np.array([], dtype=float)
        self._recent_fills: deque[float] = deque(maxlen=100)
        self._fill_history = np.array([], dtype=float)
        self._inventory_history = np.array([], dtype=float)
        self._microprice_deviation = np.array([], dtype=float)
        self._top_level_imbalance = np.array([], dtype=float)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        reset_seed = seed
        if reset_seed is None and not self._has_reset:
            reset_seed = self.initial_seed
        super().reset(seed=reset_seed)
        self._has_reset = True

        selected = self._select_daily_file(options)
        frame = pd.read_parquet(selected, columns=self.REQUIRED_COLUMNS)
        if len(frame) < 2:
            raise ValueError(f"Orderbook episode requires at least two rows: {selected}")
        numeric_columns = [*self.BOOK_COLUMNS, *self.MARKET_COLUMNS]
        self._data = {
            column: frame[column].to_numpy(dtype=float) for column in numeric_columns
        }
        self._trade_data = self._load_trade_data(frame, selected)
        mids = self._data["mid_price"]
        self._mid_returns = np.zeros(len(mids), dtype=float)
        np.divide(
            mids[1:] - mids[:-1],
            mids[:-1],
            out=self._mid_returns[1:],
            where=mids[:-1] != 0,
        )
        self._short_volatility = (
            pd.Series(self._mid_returns)
            .rolling(window=30, min_periods=2)
            .std(ddof=0)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        bid_sizes = self._data["bid_size_1"]
        ask_sizes = self._data["ask_size_1"]
        top_size = bid_sizes + ask_sizes
        self._top_level_imbalance = np.divide(
            bid_sizes - ask_sizes,
            top_size,
            out=np.zeros(len(frame), dtype=float),
            where=top_size > 0,
        )
        microprice = np.divide(
            self._data["ask_price_1"] * bid_sizes
            + self._data["bid_price_1"] * ask_sizes,
            top_size,
            out=mids.copy(),
            where=top_size > 0,
        )
        self._microprice_deviation = np.divide(
            microprice - mids,
            mids,
            out=np.zeros(len(frame), dtype=float),
            where=mids != 0,
        )
        self._fill_history = np.zeros(len(frame), dtype=float)
        self._inventory_history = np.zeros(len(frame), dtype=float)
        self.current_date = date.fromisoformat(selected.name.split("_")[1])
        self.step_count = 0
        available_steps = len(frame) - 1
        self.max_steps = min(self.episode_steps or available_steps, available_steps)
        max_start = available_steps - self.max_steps
        requested_start = options.get("start_index") if options else None
        if requested_start is not None:
            self.current_index = int(requested_start)
            if not 0 <= self.current_index <= max_start:
                raise ValueError(f"start_index must be between 0 and {max_start}")
        elif self.random_start and max_start > 0:
            self.current_index = int(self.np_random.integers(max_start + 1))
        else:
            self.current_index = 0
        self.cash = self.initial_cash
        self.inventory = 0.0
        self.equity = self.initial_cash
        self.total_fees = 0.0
        self.total_turnover = 0.0
        self.bid_fill_count = 0
        self.ask_fill_count = 0
        self.total_fill_count = 0
        self.inventory_penalty_total = 0.0
        self._recent_fills.clear()

        return self._observation(), self._info(
            action=None,
            policy_action=None,
            bid_filled=False,
            ask_filled=False,
            quote_active=False,
            size_action=1,
            bid_fill_size=0.0,
            ask_fill_size=0.0,
        )

    def step(
        self,
        action: int | list[int] | tuple[int, int] | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._data:
            raise RuntimeError("reset() must be called before step()")

        current = self.current_index
        following = current + 1
        if self.residual_continuous:
            residual_action = self._normalize_residual_action(action)
            residual = self.map_residual_action(residual_action, self.residual_variant)
            quote_action = self._base_quote_action(current)
            size_action = -1
            policy_action: Any = residual_action.copy()
            bid_requested_size = self.order_size_btc * residual["bid_size_multiplier"]
            ask_requested_size = self.order_size_btc * residual["ask_size_multiplier"]
            participates = (
                bid_requested_size > 1e-12 or ask_requested_size > 1e-12
                if self.residual_variant == "asymmetric"
                else self.np_random.random() <= residual["participation_probability"]
            )
            if participates:
                bid_quote, ask_quote = self._quote_prices(
                    quote_action,
                    current,
                    bid_spread_multiplier=residual["bid_spread_multiplier"],
                    ask_spread_multiplier=residual["ask_spread_multiplier"],
                )
            else:
                bid_quote, ask_quote = None, None
            if bid_requested_size <= 1e-12:
                bid_quote = None
            if ask_requested_size <= 1e-12:
                ask_quote = None
        else:
            quote_action, size_action, policy_action = self._normalize_action(action)
            size_multiplier = float(self.size_multipliers[size_action])
            bid_requested_size = self.order_size_btc * size_multiplier
            ask_requested_size = bid_requested_size
            residual = {
                "bid_spread_multiplier": 1.0,
                "ask_spread_multiplier": 1.0,
                "bid_size_multiplier": size_multiplier,
                "ask_size_multiplier": size_multiplier,
                "participation_probability": 1.0 if quote_action else 0.0,
            }
            bid_quote, ask_quote = self._quote_prices(quote_action, current)
        previous_equity = self.equity
        bid_fill_size = 0.0
        ask_fill_size = 0.0
        fee_paid_step = 0.0
        turnover_step = 0.0
        bid_queue_ahead = 0.0
        ask_queue_ahead = 0.0

        quote_active = bid_quote is not None or ask_quote is not None
        buy_capacity = max(0.0, self.max_inventory_btc - self.inventory)
        sell_capacity = max(0.0, self.max_inventory_btc + self.inventory)
        if self._trade_data and bid_quote is not None:
            sell_volume = float(self._trade_data["sell_volume"][following])
            min_sell_price = float(self._trade_data["min_sell_price"][following])
            bid_queue_ahead = self._visible_size("bid", bid_quote, current) * self.queue_fraction
            if sell_volume > bid_queue_ahead and np.isfinite(min_sell_price):
                if min_sell_price <= bid_quote:
                    bid_fill_size = min(
                        bid_requested_size,
                        sell_volume - bid_queue_ahead,
                        buy_capacity,
                    )
        if self._trade_data and ask_quote is not None:
            buy_volume = float(self._trade_data["buy_volume"][following])
            max_buy_price = float(self._trade_data["max_buy_price"][following])
            ask_queue_ahead = self._visible_size("ask", ask_quote, current) * self.queue_fraction
            if buy_volume > ask_queue_ahead and np.isfinite(max_buy_price):
                if max_buy_price >= ask_quote:
                    ask_fill_size = min(
                        ask_requested_size,
                        buy_volume - ask_queue_ahead,
                        sell_capacity,
                    )

        bid_filled = bid_fill_size > 1e-12
        ask_filled = ask_fill_size > 1e-12
        if bid_filled and bid_quote is not None:
            notional = bid_quote * bid_fill_size
            fee = notional * self.maker_fee
            self.cash -= notional + fee
            self.inventory += bid_fill_size
            turnover_step += notional
            fee_paid_step += fee
        if ask_filled and ask_quote is not None:
            notional = ask_quote * ask_fill_size
            fee = notional * self.maker_fee
            self.cash += notional - fee
            self.inventory -= ask_fill_size
            turnover_step += notional
            fee_paid_step += fee

        self.total_fees += fee_paid_step
        self.total_turnover += turnover_step
        self.bid_fill_count += int(bid_filled)
        self.ask_fill_count += int(ask_filled)
        self.total_fill_count += int(bid_filled) + int(ask_filled)
        fill_feature = (int(bid_filled) + int(ask_filled)) / 2.0
        self._recent_fills.append(fill_feature)
        self._fill_history[following] = fill_feature

        self.current_index = following
        self.step_count += 1
        mid_price = float(self._data["mid_price"][self.current_index])
        self.equity = self.cash + self.inventory * mid_price
        inventory_ratio = self.inventory / self.max_inventory_btc
        self._inventory_history[following] = np.clip(inventory_ratio, -1.0, 1.0)
        inventory_penalty_step = (
            self.INVENTORY_PENALTY
            * self.inventory_penalty_multiplier
            * inventory_ratio**2
        )
        downside_inventory_penalty_step = 0.0
        if self.residual_continuous:
            downside_inventory_penalty_step = (
                self.INVENTORY_PENALTY
                * self.inventory_penalty_multiplier
                * 5.0
                * max(abs(inventory_ratio) - 0.8, 0.0) ** 2
            )
        inventory_penalty_step += downside_inventory_penalty_step
        self.inventory_penalty_total += inventory_penalty_step
        raw_pnl_step = self.equity - previous_equity + fee_paid_step
        reward = self.equity - previous_equity - inventory_penalty_step
        truncated = self.step_count >= self.max_steps

        return self._observation(), float(reward), False, truncated, self._info(
            action=quote_action,
            policy_action=policy_action,
            bid_filled=bid_filled,
            ask_filled=ask_filled,
            quote_active=quote_active,
            size_action=size_action,
            bid_fill_size=bid_fill_size,
            ask_fill_size=ask_fill_size,
            fee_paid_step=fee_paid_step,
            turnover_step=turnover_step,
            inventory_penalty_step=inventory_penalty_step,
            raw_pnl_step=raw_pnl_step,
            reward=reward,
            bid_queue_ahead=bid_queue_ahead,
            ask_queue_ahead=ask_queue_ahead,
            residual=residual,
            bid_quote=bid_quote,
            ask_quote=ask_quote,
            markout_1s=self._execution_markout(
                following, bid_quote, ask_quote, bid_fill_size, ask_fill_size, 1
            ),
            markout_10s=self._execution_markout(
                following, bid_quote, ask_quote, bid_fill_size, ask_fill_size, 10
            ),
        )

    def _observation(self) -> np.ndarray:
        if self.residual_continuous:
            return self._sequence_observation()
        index = self.current_index
        mid_price = float(self._data["mid_price"][index])
        spread_bps = float(self._data["spread"][index]) / mid_price * 10_000.0
        imbalance = float(self._data["orderbook_imbalance"][index])
        trade_imbalance = 0.0
        if self._trade_data:
            buy_volume = float(self._trade_data["buy_volume"][index])
            sell_volume = float(self._trade_data["sell_volume"][index])
            total_volume = buy_volume + sell_volume
            if total_volume > 0:
                trade_imbalance = (buy_volume - sell_volume) / total_volume
        recent_fill_rate = float(np.mean(self._recent_fills)) if self._recent_fills else 0.0
        position = self.step_count / self.max_steps if self.max_steps else 0.0
        return np.array(
            [
                np.clip(self.inventory / self.max_inventory_btc, -1.0, 1.0),
                np.clip(spread_bps / self.OBSERVATION_SPREAD_SCALE_BPS, 0.0, 1.0),
                np.clip(2.0 * imbalance - 1.0, -1.0, 1.0),
                np.tanh(self._mid_returns[index] * self.OBSERVATION_RETURN_SCALE),
                np.tanh(self._short_volatility[index] * self.OBSERVATION_RETURN_SCALE),
                np.clip(trade_imbalance, -1.0, 1.0),
                np.clip(recent_fill_rate, 0.0, 1.0),
                np.clip(position, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _sequence_observation(self) -> np.ndarray:
        observation = np.zeros(
            (self.sequence_length, self.RESIDUAL_FEATURES),
            dtype=np.float32,
        )
        start = max(0, self.current_index - self.sequence_length + 1)
        indices = list(range(start, self.current_index + 1))
        offset = self.sequence_length - len(indices)
        for row, index in enumerate(indices, start=offset):
            mid_price = float(self._data["mid_price"][index])
            spread_bps = float(self._data["spread"][index]) / mid_price * 10_000.0
            trade_imbalance = self._trade_imbalance(index)
            markout_1s = self._historical_markout(index, 1)
            markout_10s = self._historical_markout(index, 10)
            observation[row] = np.array(
                [
                    np.tanh(self._mid_returns[index] * 1_000.0),
                    np.tanh(spread_bps / 20.0),
                    np.tanh(self._microprice_deviation[index] * 1_000.0),
                    np.clip(self._top_level_imbalance[index], -1.0, 1.0),
                    np.clip(
                        2.0 * self._data["orderbook_imbalance"][index] - 1.0,
                        -1.0,
                        1.0,
                    ),
                    np.clip(trade_imbalance, -1.0, 1.0),
                    np.tanh(self._short_volatility[index] * 1_000.0),
                    np.clip(self._fill_history[index], 0.0, 1.0),
                    np.tanh(markout_1s * 1_000.0),
                    np.tanh(markout_10s * 1_000.0),
                    np.clip(self._inventory_history[index], -1.0, 1.0),
                ],
                dtype=np.float32,
            )
        return observation

    def _trade_imbalance(self, index: int) -> float:
        if not self._trade_data:
            return 0.0
        buy_volume = float(self._trade_data["buy_volume"][index])
        sell_volume = float(self._trade_data["sell_volume"][index])
        total_volume = buy_volume + sell_volume
        return (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0.0

    def _historical_markout(self, index: int, lag: int) -> float:
        previous = index - lag
        if previous < 0:
            return 0.0
        previous_mid = float(self._data["mid_price"][previous])
        if previous_mid == 0:
            return 0.0
        previous_microprice = previous_mid * (
            1.0 + float(self._microprice_deviation[previous])
        )
        return (float(self._data["mid_price"][index]) - previous_microprice) / previous_mid

    def _quote_prices(
        self,
        quote_action: int,
        index: int,
        *,
        bid_spread_multiplier: float = 1.0,
        ask_spread_multiplier: float = 1.0,
    ) -> tuple[float | None, float | None]:
        if quote_action == 0:
            return None, None
        bid_multiplier, ask_multiplier = self.QUOTE_WIDTH_MULTIPLIERS[quote_action]
        bid_multiplier *= bid_spread_multiplier
        ask_multiplier *= ask_spread_multiplier
        mid_price = float(self._data["mid_price"][index])
        best_bid = float(self._data["bid_price_1"][index])
        best_ask = float(self._data["ask_price_1"][index])
        spread_bps = self.quote_spread_bps * self.quote_offset_scale
        bid_target = min(
            best_bid,
            mid_price * (1.0 - spread_bps * bid_multiplier / 20_000.0),
        )
        ask_target = max(
            best_ask,
            mid_price * (1.0 + spread_bps * ask_multiplier / 20_000.0),
        )
        bid_quote = self._available_book_price("bid", bid_target, index)
        ask_quote = self._available_book_price("ask", ask_target, index)
        inventory_ratio = self.inventory / self.max_inventory_btc
        if inventory_ratio >= self.INVENTORY_QUOTE_CUTOFF:
            bid_quote = None
        if inventory_ratio <= -self.INVENTORY_QUOTE_CUTOFF:
            ask_quote = None
        return bid_quote, ask_quote

    def _base_quote_action(self, index: int) -> int:
        observation = {
            "inventory_ratio": self.inventory / self.max_inventory_btc,
            "inventory": self.inventory,
            "volatility": float(
                np.tanh(self._short_volatility[index] * self.OBSERVATION_RETURN_SCALE)
            ),
            "orderbook_imbalance": float(
                np.clip(
                    2.0 * self._data["orderbook_imbalance"][index] - 1.0,
                    -1.0,
                    1.0,
                )
            ),
        }
        return self.base_strategy.select_action(observation)

    @staticmethod
    def neutral_residual_action(variant: str = "legacy") -> np.ndarray:
        spread_low, spread_high, size_high = RealOrderbookEnv._residual_bounds(variant)
        spread_action = 2.0 * (1.0 - spread_low) / (spread_high - spread_low) - 1.0
        size_action = 2.0 / size_high - 1.0
        return np.array(
            [spread_action, spread_action, size_action, size_action, 1.0],
            dtype=np.float32,
        )

    @staticmethod
    def map_residual_action(
        action: np.ndarray,
        variant: str = "legacy",
    ) -> dict[str, float]:
        normalized = np.asarray(action, dtype=float)
        spread_low, spread_high, size_high = RealOrderbookEnv._residual_bounds(variant)
        spread_range = spread_high - spread_low
        return {
            "bid_spread_multiplier": spread_low + 0.5 * spread_range * (normalized[0] + 1.0),
            "ask_spread_multiplier": spread_low + 0.5 * spread_range * (normalized[1] + 1.0),
            "bid_size_multiplier": 0.5 * size_high * (normalized[2] + 1.0),
            "ask_size_multiplier": 0.5 * size_high * (normalized[3] + 1.0),
            "participation_probability": (
                float(normalized[2] > -1.0 or normalized[3] > -1.0)
                if variant == "asymmetric"
                else 0.5 * (normalized[4] + 1.0)
            ),
        }

    @staticmethod
    def _residual_bounds(variant: str) -> tuple[float, float, float]:
        bounds = {
            "legacy": (0.5, 2.0, 2.0),
            "conservative": (0.5, 2.0, 1.0),
            "selective_narrow": (0.25, 1.5, 1.0),
            "asymmetric": (0.25, 2.0, 1.0),
        }
        try:
            return bounds[variant]
        except KeyError as error:
            raise ValueError(f"Unknown residual variant: {variant}") from error

    def _normalize_residual_action(
        self,
        action: int | list[int] | tuple[int, int] | np.ndarray,
    ) -> np.ndarray:
        normalized = np.asarray(action, dtype=np.float32)
        if normalized.shape != (5,) or not self.action_space.contains(normalized):
            raise ValueError("Invalid residual action; expected five values in [-1, 1]")
        return normalized

    def _available_book_price(self, side: str, target: float, index: int) -> float:
        prices = np.array(
            [self._data[f"{side}_price_{level}"][index] for level in range(1, 11)],
            dtype=float,
        )
        prices = prices[np.isfinite(prices) & (prices > 0)]
        eligible = prices[prices <= target] if side == "bid" else prices[prices >= target]
        if eligible.size == 0:
            return float(target)
        return float(np.max(eligible) if side == "bid" else np.min(eligible))

    def _visible_size(self, side: str, quote: float, index: int) -> float:
        for level in range(1, 11):
            price = float(self._data[f"{side}_price_{level}"][index])
            if np.isclose(price, quote, rtol=0.0, atol=max(abs(quote) * 1e-10, 1e-12)):
                return max(0.0, float(self._data[f"{side}_size_{level}"][index]))
        return 0.0

    def _execution_markout(
        self,
        index: int,
        bid_quote: float | None,
        ask_quote: float | None,
        bid_fill_size: float,
        ask_fill_size: float,
        horizon: int,
    ) -> float:
        total_size = bid_fill_size + ask_fill_size
        if total_size <= 1e-12:
            return 0.0
        future_index = min(index + horizon, len(self._data["mid_price"]) - 1)
        future_mid = float(self._data["mid_price"][future_index])
        weighted_markout = 0.0
        if bid_fill_size > 0.0 and bid_quote:
            weighted_markout += bid_fill_size * (future_mid - bid_quote) / bid_quote * 10_000.0
        if ask_fill_size > 0.0 and ask_quote:
            weighted_markout += ask_fill_size * (ask_quote - future_mid) / ask_quote * 10_000.0
        return weighted_markout / total_size

    def _info(
        self,
        *,
        action: int | None,
        policy_action: Any,
        bid_filled: bool,
        ask_filled: bool,
        quote_active: bool,
        size_action: int,
        bid_fill_size: float,
        ask_fill_size: float,
        fee_paid_step: float = 0.0,
        turnover_step: float = 0.0,
        inventory_penalty_step: float = 0.0,
        raw_pnl_step: float = 0.0,
        reward: float = 0.0,
        bid_queue_ahead: float = 0.0,
        ask_queue_ahead: float = 0.0,
        residual: dict[str, float] | None = None,
        bid_quote: float | None = None,
        ask_quote: float | None = None,
        markout_1s: float = 0.0,
        markout_10s: float = 0.0,
    ) -> dict[str, Any]:
        index = self.current_index
        if residual is None:
            size_multiplier = float(self.size_multipliers[size_action])
            residual = {
                "bid_spread_multiplier": 1.0,
                "ask_spread_multiplier": 1.0,
                "bid_size_multiplier": size_multiplier,
                "ask_size_multiplier": size_multiplier,
                "participation_probability": float(quote_active),
            }
        else:
            size_multiplier = 0.5 * (
                residual["bid_size_multiplier"] + residual["ask_size_multiplier"]
            )
        return {
            "equity": self.equity,
            "cash": self.cash,
            "inventory": self.inventory,
            "mid_price": float(self._data["mid_price"][index]),
            "spread": float(self._data["spread"][index]),
            "action": action,
            "policy_action": policy_action,
            "size_action": size_action,
            "size_multiplier": size_multiplier,
            "order_size_btc": self.order_size_btc * size_multiplier,
            "bid_order_size_btc": self.order_size_btc * residual["bid_size_multiplier"],
            "ask_order_size_btc": self.order_size_btc * residual["ask_size_multiplier"],
            "bid_spread_multiplier": residual["bid_spread_multiplier"],
            "ask_spread_multiplier": residual["ask_spread_multiplier"],
            "participation_probability": residual["participation_probability"],
            "bid_quote": bid_quote,
            "ask_quote": ask_quote,
            "bid_filled": bid_filled,
            "ask_filled": ask_filled,
            "bid_fill_size": bid_fill_size,
            "ask_fill_size": ask_fill_size,
            "bid_queue_ahead": bid_queue_ahead,
            "ask_queue_ahead": ask_queue_ahead,
            "quote_active": quote_active,
            "maker_fee": self.maker_fee,
            "queue_fraction": self.queue_fraction,
            "fee_paid_step": fee_paid_step,
            "total_fees": self.total_fees,
            "turnover_step": turnover_step,
            "total_turnover": self.total_turnover,
            "inventory_penalty_step": inventory_penalty_step,
            "raw_pnl_step": raw_pnl_step,
            "reward": reward,
            "bid_fill_count": self.bid_fill_count,
            "ask_fill_count": self.ask_fill_count,
            "total_fill_count": self.total_fill_count,
            "inventory_penalty_total": self.inventory_penalty_total,
            "markout_1s": markout_1s,
            "markout_10s": markout_10s,
        }

    def _select_daily_file(self, options: dict[str, Any] | None) -> Path:
        if options and "date" in options:
            selected_date = date.fromisoformat(str(options["date"]))
            for daily_date, path in self._daily_files:
                if daily_date == selected_date:
                    return path
            raise FileNotFoundError(f"No orderbook parquet found for {selected_date}")
        index = int(self.np_random.integers(len(self._daily_files)))
        return self._daily_files[index][1]

    def _normalize_action(
        self,
        action: int | list[int] | tuple[int, int] | np.ndarray,
    ) -> tuple[int, int, int]:
        if np.isscalar(action):
            normalized = np.array([int(action), 1], dtype=np.int64)
        else:
            normalized = np.asarray(action, dtype=np.int64)
        if normalized.shape != (2,) or not self.action_space.contains(normalized):
            quote_range = "0..4" if self.mandatory_quoting else "0..5"
            raise ValueError(
                f"Invalid action; expected [quote_action {quote_range}, size_action 0..2]"
            )
        policy_action = int(normalized[0])
        quote_action = policy_action + 1 if self.mandatory_quoting else policy_action
        return quote_action, int(normalized[1]), policy_action

    def _load_trade_data(
        self,
        orderbook: pd.DataFrame,
        orderbook_path: Path,
    ) -> dict[str, np.ndarray]:
        if self.trades_dir is None:
            return {}
        day = orderbook_path.name.split("_")[1]
        trade_path = self.trades_dir / f"{self.symbol}_{day}_trades_1s.parquet"
        if not trade_path.is_file():
            raise FileNotFoundError(f"Missing matching trades parquet: {trade_path}")
        trades = pd.read_parquet(
            trade_path,
            columns=[
                "timestamp",
                "buy_volume",
                "sell_volume",
                "max_buy_price",
                "min_sell_price",
            ],
        )
        trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True).dt.floor("s")
        trades = trades.drop_duplicates("timestamp", keep="last").set_index("timestamp")
        orderbook_seconds = pd.to_datetime(orderbook["timestamp"], utc=True).dt.floor("s")
        aligned = trades.reindex(orderbook_seconds)
        return {
            "buy_volume": aligned["buy_volume"].fillna(0.0).to_numpy(dtype=float),
            "sell_volume": aligned["sell_volume"].fillna(0.0).to_numpy(dtype=float),
            "max_buy_price": aligned["max_buy_price"].to_numpy(dtype=float),
            "min_sell_price": aligned["min_sell_price"].to_numpy(dtype=float),
        }

    def _find_daily_files(self) -> list[tuple[date, Path]]:
        files = []
        current = self.start_date
        while current <= self.end_date:
            path = self.data_dir / (
                f"{self.symbol}_{current.isoformat()}_orderbook_top10_1s.parquet"
            )
            if path.is_file():
                trade_path = (
                    self.trades_dir / f"{self.symbol}_{current.isoformat()}_trades_1s.parquet"
                    if self.trades_dir
                    else None
                )
                if trade_path is None or trade_path.is_file():
                    files.append((current, path))
            current += timedelta(days=1)
        return files
