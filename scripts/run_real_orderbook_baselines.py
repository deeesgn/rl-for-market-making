"""Run rule-based baselines on processed Bybit orderbook parquet files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from rl_mm.env import RealOrderbookEnv
from rl_mm.strategies import BaseStrategy, FixedSpreadStrategy, InventorySkewStrategy


@dataclass(frozen=True)
class ReplayMetrics:
    total_pnl: float
    gross_pnl_before_fees: float
    total_fees: float
    total_turnover: float
    total_reward: float
    inventory_penalty_total: float
    max_abs_inventory: float
    mean_abs_inventory: float
    final_inventory: float
    quote_rate: float
    fill_rate: float
    bid_fills: int
    ask_fills: int
    number_of_steps: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baselines on real orderbook parquet data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/bybit/orderbook/BTCUSDT"),
    )
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2025-01-07")
    parser.add_argument("--episode-steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maker-fee", type=float, default=0.0002)
    args = parser.parse_args()

    results = run_baselines(
        data_dir=args.data_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        episode_steps=args.episode_steps,
        seed=args.seed,
        maker_fee=args.maker_fee,
    )
    print_results(results)


def run_baselines(
    *,
    data_dir: Path,
    start_date: str,
    end_date: str,
    episode_steps: int | None = None,
    seed: int = 42,
    maker_fee: float = 0.0002,
) -> dict[str, ReplayMetrics]:
    dates = required_dates(data_dir, "BTCUSDT", start_date, end_date)
    strategies: list[BaseStrategy] = [
        FixedSpreadStrategy(),
        InventorySkewStrategy(threshold=0.01),
    ]
    return {
        strategy.name: evaluate_strategy(
            strategy,
            data_dir=data_dir,
            dates=dates,
            episode_steps=episode_steps,
            seed=seed,
            maker_fee=maker_fee,
        )
        for strategy in strategies
    }


def evaluate_strategy(
    strategy: BaseStrategy,
    *,
    data_dir: Path,
    dates: list[date],
    episode_steps: int | None,
    seed: int,
    maker_fee: float,
) -> ReplayMetrics:
    total_pnl = 0.0
    gross_pnl_before_fees = 0.0
    total_fees = 0.0
    total_turnover = 0.0
    total_reward = 0.0
    inventory_penalty_total = 0.0
    inventories = []
    quoted_steps = 0
    bid_fills = 0
    ask_fills = 0
    number_of_steps = 0
    final_inventory = 0.0

    for episode_index, current_date in enumerate(dates):
        day = current_date.isoformat()
        env = RealOrderbookEnv(
            data_dir=data_dir,
            start_date=day,
            end_date=day,
            episode_steps=episode_steps,
            seed=seed + episode_index,
            maker_fee=maker_fee,
        )
        observation, _ = env.reset()
        del observation
        truncated = False
        episode_reward = 0.0
        final_equity = env.initial_cash
        while not truncated:
            action = strategy.select_action({"inventory": np.float32(env.inventory)})
            _, reward, _, truncated, info = env.step(action)
            episode_reward += reward
            final_equity = float(info["equity"])
            inventories.append(abs(float(info["inventory"])))
            quoted_steps += int(info["quote_active"])
            number_of_steps += 1
        episode_pnl = final_equity - env.initial_cash
        total_pnl += episode_pnl
        gross_pnl_before_fees += episode_pnl + env.total_fees
        total_fees += env.total_fees
        total_turnover += env.total_turnover
        total_reward += episode_reward
        inventory_penalty_total += env.inventory_penalty_total
        bid_fills += env.bid_fill_count
        ask_fills += env.ask_fill_count
        final_inventory = env.inventory

    return ReplayMetrics(
        total_pnl=total_pnl,
        gross_pnl_before_fees=gross_pnl_before_fees,
        total_fees=total_fees,
        total_turnover=total_turnover,
        total_reward=total_reward,
        inventory_penalty_total=inventory_penalty_total,
        max_abs_inventory=max(inventories, default=0.0),
        mean_abs_inventory=float(np.mean(inventories)) if inventories else 0.0,
        final_inventory=final_inventory,
        quote_rate=quoted_steps / number_of_steps if number_of_steps else 0.0,
        fill_rate=(bid_fills + ask_fills) / number_of_steps if number_of_steps else 0.0,
        bid_fills=bid_fills,
        ask_fills=ask_fills,
        number_of_steps=number_of_steps,
    )


def required_dates(data_dir: Path, symbol: str, start_date: str, end_date: str) -> list[date]:
    current = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    dates = []
    missing = []
    while current <= end:
        path = data_dir / f"{symbol}_{current.isoformat()}_orderbook_top10_1s.parquet"
        if path.is_file():
            dates.append(current)
        else:
            missing.append(current.isoformat())
        current += timedelta(days=1)
    if missing:
        raise FileNotFoundError("Missing orderbook parquet dates: " + ", ".join(missing))
    return dates


def print_results(results: dict[str, ReplayMetrics]) -> None:
    for name, metrics in results.items():
        print(f"strategy: {name}")
        print(f"  total_pnl: {metrics.total_pnl:.4f}")
        print(f"  gross_pnl_before_fees: {metrics.gross_pnl_before_fees:.4f}")
        print(f"  total_fees: {metrics.total_fees:.4f}")
        print(f"  total_turnover: {metrics.total_turnover:.4f}")
        print(f"  total_reward: {metrics.total_reward:.4f}")
        print(f"  inventory_penalty_total: {metrics.inventory_penalty_total:.4f}")
        print(f"  max_abs_inventory: {metrics.max_abs_inventory:.4f}")
        print(f"  mean_abs_inventory: {metrics.mean_abs_inventory:.4f}")
        print(f"  final_inventory: {metrics.final_inventory:.4f}")
        print(f"  quote_rate: {metrics.quote_rate:.4f}")
        print(f"  fill_rate: {metrics.fill_rate:.4f}")
        print(f"  bid_fills: {metrics.bid_fills}")
        print(f"  ask_fills: {metrics.ask_fills}")
        print(f"  number_of_steps: {metrics.number_of_steps}")


if __name__ == "__main__":
    main()
