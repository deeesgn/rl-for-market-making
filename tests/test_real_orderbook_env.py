from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rl_mm.env import RealOrderbookEnv
from scripts.run_real_orderbook_baselines import run_baselines

REQUIRED_INFO = {
    "equity",
    "cash",
    "inventory",
    "mid_price",
    "spread",
    "action",
    "bid_filled",
    "ask_filled",
    "quote_active",
    "fee_paid_step",
    "total_fees",
    "turnover_step",
    "total_turnover",
    "inventory_penalty_step",
    "raw_pnl_step",
    "reward",
}


def test_real_orderbook_env_reset_and_observation_shape(tmp_path: Path) -> None:
    write_fake_orderbook(tmp_path)
    env = make_env(tmp_path)

    observation, info = env.reset()

    assert observation.shape == (8,)
    assert env.observation_space.contains(observation)
    assert float(observation[0]) == 0.0
    assert info["equity"] == 10_000.0
    assert REQUIRED_INFO <= info.keys()


def test_real_orderbook_env_step_returns_required_info(tmp_path: Path) -> None:
    write_fake_orderbook(tmp_path)
    env = make_env(tmp_path, episode_steps=1)
    env.reset()

    observation, reward, terminated, truncated, info = env.step(1)

    assert observation.shape == (8,)
    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is True
    assert info["action"] == 1
    assert info["quote_active"] is True
    assert REQUIRED_INFO <= info.keys()
    assert info["total_fees"] >= 0.0
    assert info["total_turnover"] >= 0.0
    assert np.isclose(
        info["equity"],
        info["cash"] + info["inventory"] * info["mid_price"],
    )
    assert np.isclose(
        reward,
        info["raw_pnl_step"] - info["fee_paid_step"] - info["inventory_penalty_step"],
    )


def test_real_orderbook_env_respects_inventory_limit(tmp_path: Path) -> None:
    write_fake_orderbook(tmp_path)
    env = make_env(tmp_path, max_inventory_btc=0.002, order_size_btc=0.001)
    env.reset()

    truncated = False
    while not truncated:
        _, _, _, truncated, info = env.step(1)
        assert abs(float(info["inventory"])) <= 0.002 + 1e-12

    assert abs(env.inventory) <= 0.002 + 1e-12


def test_no_quote_keeps_cash_inventory_and_fills_unchanged(tmp_path: Path) -> None:
    write_fake_orderbook(tmp_path)
    env = make_env(tmp_path, episode_steps=3)
    env.reset()

    for _ in range(3):
        _, _, _, _, info = env.step(0)
        assert info["bid_filled"] is False
        assert info["ask_filled"] is False
        assert info["turnover_step"] == 0.0
        assert info["fee_paid_step"] == 0.0

    assert env.cash == env.initial_cash
    assert env.inventory == 0.0
    assert env.bid_fill_count == 0
    assert env.ask_fill_count == 0
    assert env.total_fill_count == 0
    assert env.total_fees == 0.0
    assert env.total_turnover == 0.0


def test_zero_maker_fee_keeps_total_fees_zero(tmp_path: Path) -> None:
    write_fake_orderbook(tmp_path)
    env = make_env(tmp_path, maker_fee=0.0, episode_steps=2)
    env.reset()

    env.step(1)
    _, _, _, _, info = env.step(1)

    assert info["total_turnover"] >= 0.0
    assert info["total_fees"] == 0.0


def test_real_baseline_script_runs_on_fake_data(tmp_path: Path) -> None:
    write_fake_orderbook(tmp_path)

    results = run_baselines(
        data_dir=tmp_path,
        start_date="2025-01-01",
        end_date="2025-01-01",
        episode_steps=3,
        seed=7,
    )

    assert set(results) == {"fixed_spread", "inventory_skew"}
    assert results["fixed_spread"].number_of_steps == 3
    assert results["inventory_skew"].number_of_steps == 3
    assert results["fixed_spread"].quote_rate == 1.0
    for metrics in results.values():
        assert np.isclose(
            metrics.total_pnl,
            metrics.gross_pnl_before_fees - metrics.total_fees,
        )
        assert np.isclose(
            metrics.total_reward,
            metrics.total_pnl - metrics.inventory_penalty_total,
        )


def make_env(data_dir: Path, **overrides) -> RealOrderbookEnv:
    return RealOrderbookEnv(
        data_dir=data_dir,
        start_date="2025-01-01",
        end_date="2025-01-01",
        seed=7,
        **overrides,
    )


def write_fake_orderbook(data_dir: Path) -> Path:
    timestamps = pd.date_range("2025-01-01", periods=5, freq="s", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "bid_price_1": [100.0, 98.0, 96.0, 94.0, 92.0],
            "ask_price_1": [101.0, 99.0, 97.0, 95.0, 93.0],
            "mid_price": [100.5, 98.5, 96.5, 94.5, 92.5],
            "spread": [1.0] * 5,
            "orderbook_imbalance": [0.5] * 5,
        }
    )
    for level in range(1, 11):
        if level > 1:
            frame[f"bid_price_{level}"] = frame["bid_price_1"] - (level - 1) * 0.5
            frame[f"ask_price_{level}"] = frame["ask_price_1"] + (level - 1) * 0.5
        frame[f"bid_size_{level}"] = 0.001
        frame[f"ask_size_{level}"] = 0.001
    path = data_dir / "BTCUSDT_2025-01-01_orderbook_top10_1s.parquet"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path
