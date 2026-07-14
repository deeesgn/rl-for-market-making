from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl_mm.data.trades import aggregate_trade_chunks
from rl_mm.env import RealOrderbookEnv
from rl_mm.strategies import FixedSpreadStrategy, InventorySkewStrategy
from scripts.run_real_experiment import (
    SEARCH_CANDIDATES,
    BaselineParameters,
    DatasetSplit,
    EvaluationSummary,
    MeanStd,
    atomic_copy_model,
    baseline_selector,
    build_evaluation_windows,
    build_residual_test_windows,
    chronological_split,
    cleanup_rejected_models,
    collect_behavior_cloning_batch,
    evaluate_actor,
    initial_search_state,
    load_models,
    load_search_state,
    model_path_for_seed,
    ppo_selector,
    promote_candidates,
    residual_validation_result,
    resolve_model_path,
    run_search_final_test,
    save_search_state,
    search_model_path,
    search_result_score,
    select_best_baseline,
    select_best_seed,
    select_residual_seed,
    select_residual_variant,
    train_ppo_models,
    train_residual_sac,
    train_search_candidate,
    validation_score_components,
)


def test_trade_aggregation() -> None:
    chunks = [
        pd.DataFrame(
            {
                "timestamp": [1704067200.1, 1704067200.8],
                "side": ["Buy", "Sell"],
                "size": [1.5, 2.0],
                "price": [101.0, 99.0],
            }
        ),
        pd.DataFrame(
            {
                "timestamp": [1704067200.9, 1704067201.2],
                "side": ["Buy", "Sell"],
                "size": [0.5, 3.0],
                "price": [102.0, 98.0],
            }
        ),
    ]

    result = aggregate_trade_chunks(chunks)

    assert len(result) == 2
    assert result.loc[0, "buy_volume"] == 2.0
    assert result.loc[0, "sell_volume"] == 2.0
    assert result.loc[0, "max_buy_price"] == 102.0
    assert result.loc[0, "min_sell_price"] == 99.0
    assert result.loc[1, "sell_volume"] == 3.0


def test_trade_driven_bid_fill(tmp_path: Path) -> None:
    orderbook_dir, trades_dir = write_market_day(tmp_path, sell_at_step=1)
    env = make_trade_env(orderbook_dir, trades_dir)
    env.reset()

    _, _, _, _, info = env.step([1, 1])

    assert info["bid_filled"] is True
    assert info["ask_filled"] is False
    assert np.isclose(info["inventory"], 0.001)


def test_trade_driven_ask_fill(tmp_path: Path) -> None:
    orderbook_dir, trades_dir = write_market_day(tmp_path, buy_at_step=1)
    env = make_trade_env(orderbook_dir, trades_dir)
    env.reset()

    _, _, _, _, info = env.step([1, 1])

    assert info["bid_filled"] is False
    assert bool(info["ask_filled"]) is True
    assert np.isclose(info["inventory"], -0.001)


def test_no_trades_means_no_fill(tmp_path: Path) -> None:
    orderbook_dir, trades_dir = write_market_day(tmp_path)
    env = make_trade_env(orderbook_dir, trades_dir)
    env.reset()

    _, _, _, _, info = env.step([1, 2])

    assert info["bid_filled"] is False
    assert info["ask_filled"] is False
    assert info["total_turnover"] == 0.0


def test_size_actions_change_order_size(tmp_path: Path) -> None:
    orderbook_dir, trades_dir = write_market_day(tmp_path, sell_at_step=1, trade_volume=1.0)
    inventories = []
    for size_action in range(3):
        env = make_trade_env(orderbook_dir, trades_dir)
        env.reset()
        _, _, _, _, info = env.step([1, size_action])
        inventories.append(info["inventory"])

    np.testing.assert_allclose(inventories, [0.0005, 0.001, 0.002])


def test_continuous_residual_actions_map_to_requested_ranges() -> None:
    low = RealOrderbookEnv.map_residual_action(np.full(5, -1.0, dtype=np.float32))
    high = RealOrderbookEnv.map_residual_action(np.ones(5, dtype=np.float32))

    assert low == {
        "bid_spread_multiplier": 0.5,
        "ask_spread_multiplier": 0.5,
        "bid_size_multiplier": 0.0,
        "ask_size_multiplier": 0.0,
        "participation_probability": 0.0,
    }
    assert high == {
        "bid_spread_multiplier": 2.0,
        "ask_spread_multiplier": 2.0,
        "bid_size_multiplier": 2.0,
        "ask_size_multiplier": 2.0,
        "participation_probability": 1.0,
    }


def test_zero_residual_bid_size_creates_no_bid_order(tmp_path: Path) -> None:
    orderbook_dir, trades_dir = write_market_day(tmp_path, sell_at_step=1)
    env = make_trade_env(orderbook_dir, trades_dir, residual_continuous=True)
    env.reset()
    action = RealOrderbookEnv.neutral_residual_action()
    action[2] = -1.0

    _, _, _, _, info = env.step(action)

    assert info["bid_quote"] is None
    assert info["bid_order_size_btc"] == 0.0
    assert info["bid_filled"] is False


def test_neutral_residual_keeps_base_quote_unchanged(tmp_path: Path) -> None:
    orderbook_dir, trades_dir = write_market_day(tmp_path)
    env = make_trade_env(
        orderbook_dir,
        trades_dir,
        residual_continuous=True,
        quote_spread_bps=15.0,
        base_spread_bps=15.0,
        base_imbalance_filter=True,
    )
    env.reset()
    base_action = env._base_quote_action(env.current_index)
    expected_bid, expected_ask = env._quote_prices(base_action, env.current_index)

    _, _, _, _, info = env.step(RealOrderbookEnv.neutral_residual_action())

    assert info["action"] == base_action
    assert info["bid_quote"] == expected_bid
    assert info["ask_quote"] == expected_ask
    assert info["participation_probability"] == 1.0


def test_asymmetric_residual_supports_one_sided_quoting(tmp_path: Path) -> None:
    orderbook_dir, trades_dir = write_market_day(tmp_path, buy_at_step=1)
    env = make_trade_env(
        orderbook_dir,
        trades_dir,
        residual_continuous=True,
        residual_variant="asymmetric",
    )
    env.reset()
    action = env.neutral_residual_action("asymmetric")
    action[2] = -1.0

    _, _, _, _, info = env.step(action)

    assert info["bid_quote"] is None
    assert info["ask_quote"] is not None
    assert info["bid_filled"] is False
    assert bool(info["ask_filled"]) is True
    assert info["participation_probability"] == 1.0


def test_maker_fee_is_configurable_and_zero_fee_has_no_deduction(tmp_path: Path) -> None:
    orderbook_dir, trades_dir = write_market_day(tmp_path, sell_at_step=1)
    fee_env = make_trade_env(orderbook_dir, trades_dir, maker_fee=0.001)
    fee_env.reset()
    _, _, _, _, fee_info = fee_env.step([1, 1])

    zero_env = make_trade_env(orderbook_dir, trades_dir, maker_fee=0.0)
    zero_env.reset()
    _, _, _, _, zero_info = zero_env.step([1, 1])

    assert fee_info["fee_paid_step"] > 0.0
    assert fee_info["maker_fee"] == 0.001
    assert zero_info["bid_filled"] is True
    assert zero_info["fee_paid_step"] == 0.0
    assert zero_info["total_fees"] == 0.0


def test_mandatory_mode_has_only_five_quote_actions(tmp_path: Path) -> None:
    orderbook_dir, trades_dir = write_market_day(tmp_path)
    env = make_trade_env(
        orderbook_dir,
        trades_dir,
        mandatory_quoting=True,
    )
    env.reset()

    assert env.action_space.nvec.tolist() == [5, 3]
    _, _, _, _, info = env.step([0, 1])
    assert info["policy_action"] == 0
    assert info["action"] == 1
    assert info["quote_active"] is True


def test_reward_is_equity_change_minus_inventory_penalty(tmp_path: Path) -> None:
    orderbook_dir, trades_dir = write_market_day(tmp_path, sell_at_step=1)
    env = make_trade_env(orderbook_dir, trades_dir, maker_fee=0.0)
    env.reset()
    previous_equity = env.equity

    _, reward, _, _, info = env.step([1, 1])

    expected = info["equity"] - previous_equity - info["inventory_penalty_step"]
    assert np.isclose(reward, expected)


def test_visible_queue_prevents_premature_fill(tmp_path: Path) -> None:
    orderbook_dir, trades_dir = write_market_day(
        tmp_path,
        sell_at_step=1,
        trade_volume=0.005,
        visible_size=0.01,
    )
    env = make_trade_env(orderbook_dir, trades_dir, queue_fraction=0.5)
    env.reset()

    _, _, _, _, info = env.step([1, 1])

    assert info["bid_queue_ahead"] == 0.005
    assert info["bid_filled"] is False


def test_queue_excess_produces_partial_fill(tmp_path: Path) -> None:
    orderbook_dir, trades_dir = write_market_day(
        tmp_path,
        sell_at_step=1,
        trade_volume=0.0054,
        visible_size=0.01,
    )
    env = make_trade_env(orderbook_dir, trades_dir, queue_fraction=0.5)
    env.reset()

    _, _, _, _, info = env.step([1, 1])

    assert info["bid_filled"] is True
    assert np.isclose(info["bid_fill_size"], 0.0004)


def test_chronological_split_never_shuffles() -> None:
    dates = [date(2025, 1, 1) + timedelta(days=index) for index in range(20)]

    split = chronological_split(dates)

    assert split.train == tuple(dates[:15])
    assert split.validation == tuple(dates[15:17])
    assert split.test == tuple(dates[17:])


def test_model_paths_are_separate_per_seed(tmp_path: Path) -> None:
    base = tmp_path / "models" / "real_ppo.zip"

    assert model_path_for_seed(base, 42).name == "real_ppo_seed42.zip"
    assert model_path_for_seed(base, 100).name == "real_ppo_seed100.zip"
    assert model_path_for_seed(base, 200).name == "real_ppo_seed200.zip"


def test_best_seed_uses_validation_reward() -> None:
    assert select_best_seed({42: -3.0, 100: 1.5, 200: 0.5}) == 100


def test_baseline_selection_uses_supplied_validation_rewards() -> None:
    first = BaselineParameters(2.5, 0.25, False, False)
    second = BaselineParameters(10.0, 0.75, True, True)

    assert select_best_baseline({first: -1.0, second: 2.0}) == second


def test_zero_fee_models_do_not_share_fee_based_paths(tmp_path: Path) -> None:
    fee_base = tmp_path / "models" / "real_ppo.zip"
    zero_base = resolve_model_path(
        None,
        maker_fee=0.0,
        mandatory_quoting=True,
    )

    assert zero_base.name == "real_ppo_zero_fee.zip"
    assert model_path_for_seed(zero_base, 42).name == "real_ppo_zero_fee_seed42.zip"
    assert model_path_for_seed(fee_base, 42) != model_path_for_seed(zero_base, 42)


def test_resumed_zero_fee_training_does_not_overwrite_fee_model(tmp_path: Path) -> None:
    fee_base = tmp_path / "models" / "real_ppo.zip"
    zero_base = tmp_path / "models" / "real_ppo_zero_fee.zip"
    fee_model = model_path_for_seed(fee_base, 42)
    zero_model = model_path_for_seed(zero_base, 42)
    fee_model.parent.mkdir(parents=True)
    fee_model.write_bytes(b"fee model")
    zero_model.write_bytes(b"zero fee model")

    train_ppo_models(
        orderbook_dir=tmp_path / "missing-orderbook",
        trades_dir=tmp_path / "missing-trades",
        train_dates=(date(2025, 1, 1),),
        timesteps=16,
        seeds=[42],
        model_path=zero_base,
        maker_fee=0.0,
        mandatory_quoting=True,
    )

    assert fee_model.read_bytes() == b"fee model"
    assert zero_model.read_bytes() == b"zero fee model"


def test_search_ranking_formula_and_stage_promotion() -> None:
    values = {
        "mean_validation_pnl": 3.0,
        "std_validation_pnl": 2.0,
        "mean_max_drawdown": 5.0,
    }
    assert search_result_score(values) == 2.0

    results = {
        "validation_winner": {"eligible": True, "score": 2.0},
        "runner_up": {"eligible": True, "score": 1.0},
        "inactive": {"eligible": False, "score": None},
    }
    assert promote_candidates(results, 2) == ["validation_winner", "runner_up"]


def test_completed_search_candidate_is_reused(tmp_path: Path) -> None:
    candidate = SEARCH_CANDIDATES[0]
    state = initial_search_state()
    state["tasks"] = {
        f"{candidate.name}:seed42": {"timesteps": 60_000, "status": "complete"}
    }
    model_path = search_model_path(tmp_path, candidate.name, 42)
    model_path.write_bytes(b"completed")

    reused = train_search_candidate(
        candidate,
        seed=42,
        target_timesteps=60_000,
        orderbook_dir=tmp_path / "missing-orderbook",
        trades_dir=tmp_path / "missing-trades",
        train_dates=(date(2025, 1, 1),),
        search_dir=tmp_path,
        state=state,
        state_path=tmp_path / "state.json",
        deadline=float("inf"),
    )

    assert reused == model_path
    assert reused.read_bytes() == b"completed"


def test_search_state_resumes_from_atomic_checkpoint(tmp_path: Path) -> None:
    state_path = tmp_path / "search_state.json"
    state = initial_search_state()
    state["status"] = "paused"
    state["tasks"] = {
        "base:seed42": {"timesteps": 50_000, "status": "checkpointed"}
    }

    save_search_state(state_path, state)
    resumed = load_search_state(state_path)

    assert resumed["status"] == "running"
    assert resumed["tasks"]["base:seed42"]["timesteps"] == 50_000
    assert not state_path.with_suffix(".json.tmp").exists()


def test_rejected_search_models_are_cleaned_up(tmp_path: Path) -> None:
    rejected = search_model_path(tmp_path, "rejected", 42)
    finalist = search_model_path(tmp_path, "finalist", 42)
    rejected.write_bytes(b"remove")
    finalist.write_bytes(b"keep")

    cleanup_rejected_models(tmp_path, ["rejected"])

    assert not rejected.exists()
    assert finalist.read_bytes() == b"keep"


def test_final_test_cannot_access_data_before_selection(tmp_path: Path) -> None:
    split = DatasetSplit(
        train=(date(2025, 1, 1),),
        validation=(date(2025, 1, 2),),
        test=(date(2025, 1, 3),),
    )

    with pytest.raises(RuntimeError, match="before final selection"):
        run_search_final_test(
            state=initial_search_state(),
            state_path=tmp_path / "state.json",
            orderbook_dir=tmp_path / "missing-orderbook",
            trades_dir=tmp_path / "missing-trades",
            split=split,
            final_model_path=tmp_path / "missing.zip",
        )


def test_final_search_model_is_copied_atomically(tmp_path: Path) -> None:
    source = tmp_path / "candidate.zip"
    destination = tmp_path / "best.zip"
    source.write_bytes(b"model bytes")

    atomic_copy_model(source, destination)

    assert destination.read_bytes() == b"model bytes"
    assert not (tmp_path / "best.tmp.zip").exists()


def test_behavior_cloning_samples_train_dates_only(tmp_path: Path) -> None:
    orderbook_dir, trades_dir, dates = write_market_range(tmp_path, days=7)
    split = chronological_split(dates)
    parameters = BaselineParameters(15.0, 0.25, False, True)

    observations, actions, used_dates = collect_behavior_cloning_batch(
        orderbook_dir=orderbook_dir,
        trades_dir=trades_dir,
        train_dates=split.train,
        base_parameters=parameters,
        sample_count=8,
        seed=42,
    )

    assert observations.shape == (8, 30, 11)
    assert actions.shape == (8, 5)
    assert set(used_dates) <= set(split.train)
    assert not set(used_dates) & set(split.validation + split.test)


def test_residual_activity_gates_are_relative_to_baseline() -> None:
    baseline = make_evaluation_summary(
        pnl=0.1,
        pnl_std=0.1,
        quote_rate=1.0,
        fill_rate=0.0007,
    )
    eligible = make_evaluation_summary(
        pnl=1.0,
        quote_rate=0.5,
        fill_rate=0.00035,
    )
    inactive = make_evaluation_summary(
        pnl=100.0,
        quote_rate=0.49,
        fill_rate=0.00034,
    )

    eligible_result = residual_validation_result(
        eligible,
        baseline_summary=baseline,
    )
    inactive_result = residual_validation_result(
        inactive,
        baseline_summary=baseline,
    )

    assert eligible_result["eligible"] is True
    assert eligible_result["quote_threshold"] == 0.5
    assert eligible_result["fill_threshold"] == 0.00035
    assert inactive_result["eligible"] is False


def test_residual_ranking_uses_requested_score_formula() -> None:
    summary = make_evaluation_summary(pnl=3.0, pnl_std=2.0, drawdown=5.0)

    components = validation_score_components(summary)

    assert components == {
        "mean_validation_pnl": 3.0,
        "std_validation_pnl": 2.0,
        "mean_max_drawdown": 5.0,
        "score": 2.0,
    }


def test_previous_ppo_is_not_a_residual_hard_gate() -> None:
    baseline = make_evaluation_summary(pnl=0.5, pnl_std=0.1)
    candidate = make_evaluation_summary(pnl=1.0, pnl_std=0.1)

    result = residual_validation_result(candidate, baseline_summary=baseline)

    assert result["advances"] is True
    assert "beats_previous_ppo" not in result


def test_residual_selection_uses_validation_summaries_only() -> None:
    baseline = make_evaluation_summary(pnl=0.5, pnl_std=0.1)
    validation = {
        42: make_evaluation_summary(pnl=0.8, pnl_std=0.4),
        100: make_evaluation_summary(pnl=1.0, pnl_std=0.1),
    }

    selected = select_residual_seed(
        validation,
        baseline_summary=baseline,
    )

    assert selected == 100


def test_failed_screening_does_not_select_variant() -> None:
    baseline = make_evaluation_summary(pnl=0.5, pnl_std=0.1)
    failed = {
        "conservative": make_evaluation_summary(pnl=-0.1),
        "selective_narrow": make_evaluation_summary(pnl=0.0),
        "asymmetric": make_evaluation_summary(
            pnl=10.0,
            quote_rate=0.0,
            fill_rate=0.0,
        ),
    }

    assert select_residual_variant(failed, baseline_summary=baseline) is None


def test_residual_test_data_is_blocked_before_promotion(tmp_path: Path) -> None:
    split = DatasetSplit(
        train=(date(2025, 1, 1),),
        validation=(date(2025, 1, 2),),
        test=(date(2025, 1, 3),),
    )

    with pytest.raises(RuntimeError, match="before residual promotion"):
        build_residual_test_windows(
            orderbook_dir=tmp_path / "missing",
            split=split,
            promotion_complete=False,
        )


def test_residual_final_model_is_saved_atomically(tmp_path: Path) -> None:
    source = tmp_path / "residual_seed42.zip"
    destination = tmp_path / "real_residual_sac_zero_fee_best.zip"
    source.write_bytes(b"residual model")

    atomic_copy_model(source, destination)

    assert destination.read_bytes() == b"residual model"
    assert not destination.with_name("real_residual_sac_zero_fee_best.tmp.zip").exists()


def test_completed_model_is_not_overwritten(tmp_path: Path) -> None:
    base = tmp_path / "models" / "real_ppo.zip"
    completed = model_path_for_seed(base, 42)
    completed.parent.mkdir(parents=True)
    completed.write_bytes(b"completed model")

    paths = train_ppo_models(
        orderbook_dir=tmp_path / "missing-orderbook",
        trades_dir=tmp_path / "missing-trades",
        train_dates=(date(2025, 1, 1),),
        timesteps=16,
        seeds=[42],
        model_path=base,
    )

    assert paths[42] == completed
    assert completed.read_bytes() == b"completed model"


def test_environment_reset_and_step_with_trades(tmp_path: Path) -> None:
    orderbook_dir, trades_dir = write_market_day(tmp_path, sell_at_step=1)
    env = make_trade_env(orderbook_dir, trades_dir)

    observation, _ = env.reset()
    next_observation, reward, terminated, truncated, info = env.step([2, 1])

    assert observation.shape == (8,)
    assert next_observation.shape == (8,)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    assert info["size_action"] == 1


def test_short_ppo_training_and_identical_evaluation_windows(tmp_path: Path) -> None:
    orderbook_dir, trades_dir, dates = write_market_range(tmp_path, days=7)
    split = chronological_split(dates)
    model_path = tmp_path / "models" / "real_ppo.zip"

    trained = train_ppo_models(
        orderbook_dir=orderbook_dir,
        trades_dir=trades_dir,
        train_dates=split.train,
        timesteps=16,
        seeds=[42],
        model_path=model_path,
        episode_steps=8,
    )
    windows = build_evaluation_windows(
        orderbook_dir=orderbook_dir,
        symbol="BTCUSDT",
        dates=split.test,
        seeds=[42],
        episode_steps=8,
    )
    model = load_models(model_path, [42])[42]
    selectors = {
        "fixed_spread": baseline_selector(FixedSpreadStrategy()),
        "inventory_skew": baseline_selector(InventorySkewStrategy(threshold=0.01)),
        "ppo_seed42": ppo_selector(model),
    }
    results = {
        name: evaluate_actor(
            selector,
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            windows=windows,
        )
        for name, selector in selectors.items()
    }

    assert trained[42].is_file()
    assert trained[42].name == "real_ppo_seed42.zip"
    assert set(results) == {"fixed_spread", "inventory_skew", "ppo_seed42"}
    window_ids = {metrics.window_ids for metrics in results.values()}
    assert window_ids == {tuple(window.identifier for window in windows)}


def test_short_residual_sac_training_with_cloning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orderbook_dir, trades_dir, dates = write_market_range(tmp_path, days=7)
    split = chronological_split(dates)
    monkeypatch.chdir(tmp_path)

    path, imitation = train_residual_sac(
        orderbook_dir=orderbook_dir,
        trades_dir=trades_dir,
        train_dates=split.train,
        base_parameters=BaselineParameters(15.0, 0.25, False, True),
        seed=42,
        residual_variant="conservative",
        target_timesteps=16,
        bc_samples=8,
        screening=True,
    )

    assert path.is_file()
    assert imitation is not None
    assert imitation[1] < imitation[0]
    assert not path.with_name(f"{path.stem}.tmp.zip").exists()


def make_trade_env(
    orderbook_dir: Path,
    trades_dir: Path,
    **overrides,
) -> RealOrderbookEnv:
    return RealOrderbookEnv(
        data_dir=orderbook_dir,
        trades_dir=trades_dir,
        start_date="2025-01-01",
        end_date="2025-01-01",
        episode_steps=4,
        seed=7,
        **overrides,
    )


def make_evaluation_summary(
    *,
    pnl: float,
    pnl_std: float = 0.1,
    quote_rate: float = 0.2,
    fill_rate: float = 0.01,
    drawdown: float = 0.1,
) -> EvaluationSummary:
    zero = MeanStd(0.0, 0.0)
    return EvaluationSummary(
        total_pnl=MeanStd(pnl, pnl_std),
        gross_pnl_before_fees=MeanStd(pnl, pnl_std),
        total_fees=zero,
        total_reward=MeanStd(pnl, pnl_std),
        sharpe_ratio=zero,
        maximum_drawdown=MeanStd(drawdown, 0.0),
        max_abs_inventory=MeanStd(0.001, 0.0),
        mean_abs_inventory=MeanStd(0.0005, 0.0),
        final_inventory=zero,
        quote_rate=MeanStd(quote_rate, 0.0),
        fill_rate=MeanStd(fill_rate, 0.0),
        total_turnover=zero,
        markout_1s=zero,
        markout_10s=zero,
        profitable_episode_percentage=MeanStd(100.0 if pnl > 0 else 0.0, 0.0),
        median_total_pnl=pnl,
        maximum_observed_inventory=0.001,
        window_ids=(),
    )


def write_market_range(tmp_path: Path, *, days: int) -> tuple[Path, Path, list[date]]:
    dates = []
    for index in range(days):
        current = date(2025, 1, 1) + timedelta(days=index)
        dates.append(current)
        write_market_day(
            tmp_path,
            current_date=current,
            sell_at_step=1,
            buy_at_step=2,
        )
    return tmp_path / "orderbook", tmp_path / "trades", dates


def write_market_day(
    tmp_path: Path,
    *,
    current_date: date = date(2025, 1, 1),
    sell_at_step: int | None = None,
    buy_at_step: int | None = None,
    trade_volume: float = 0.01,
    visible_size: float = 0.002,
) -> tuple[Path, Path]:
    rows = 20
    timestamps = pd.date_range(current_date.isoformat(), periods=rows, freq="s", tz="UTC")
    orderbook = pd.DataFrame(
        {
            "timestamp": timestamps,
            "bid_price_1": [100.0] * rows,
            "ask_price_1": [101.0] * rows,
            "mid_price": [100.5] * rows,
            "spread": [1.0] * rows,
            "orderbook_imbalance": [0.5] * rows,
        }
    )
    for level in range(1, 11):
        if level > 1:
            orderbook[f"bid_price_{level}"] = 100.0 - (level - 1) * 0.5
            orderbook[f"ask_price_{level}"] = 101.0 + (level - 1) * 0.5
        orderbook[f"bid_size_{level}"] = visible_size
        orderbook[f"ask_size_{level}"] = visible_size
    trades = pd.DataFrame(
        {
            "timestamp": timestamps,
            "buy_volume": [0.0] * rows,
            "sell_volume": [0.0] * rows,
            "max_buy_price": [np.nan] * rows,
            "min_sell_price": [np.nan] * rows,
        }
    )
    if sell_at_step is not None:
        trades.loc[sell_at_step, ["sell_volume", "min_sell_price"]] = [
            trade_volume,
            100.0,
        ]
    if buy_at_step is not None:
        trades.loc[buy_at_step, ["buy_volume", "max_buy_price"]] = [
            trade_volume,
            101.0,
        ]

    day = current_date.isoformat()
    orderbook_dir = tmp_path / "orderbook"
    trades_dir = tmp_path / "trades"
    orderbook_dir.mkdir(parents=True, exist_ok=True)
    trades_dir.mkdir(parents=True, exist_ok=True)
    orderbook.to_parquet(
        orderbook_dir / f"BTCUSDT_{day}_orderbook_top10_1s.parquet",
        index=False,
    )
    trades.to_parquet(trades_dir / f"BTCUSDT_{day}_trades_1s.parquet", index=False)
    return orderbook_dir, trades_dir
    cleanup_rejected_models,
    initial_search_state,
    load_search_state,
