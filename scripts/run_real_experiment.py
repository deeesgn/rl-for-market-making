"""Prepare trades, train PPO models, and evaluate the full real-data experiment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass, field, fields
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow.parquet as pq
import torch
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from rl_mm.data.trades import prepare_trade_range
from rl_mm.env import RealOrderbookEnv
from rl_mm.strategies import BaseStrategy, FixedSpreadStrategy, InventorySkewStrategy


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[date, ...]
    validation: tuple[date, ...]
    test: tuple[date, ...]


@dataclass(frozen=True)
class EvaluationWindow:
    day: date
    start_index: int
    steps: int
    seed: int

    @property
    def identifier(self) -> str:
        return f"{self.day.isoformat()}:{self.start_index}:{self.steps}:{self.seed}"


@dataclass(frozen=True)
class EpisodeMetrics:
    total_pnl: float
    gross_pnl_before_fees: float
    total_fees: float
    total_reward: float
    sharpe_ratio: float
    maximum_drawdown: float
    max_abs_inventory: float
    mean_abs_inventory: float
    final_inventory: float
    quote_rate: float
    fill_rate: float
    total_turnover: float
    markout_1s: float
    markout_10s: float
    profitable_episode_percentage: float


@dataclass(frozen=True)
class MeanStd:
    mean: float
    std: float


@dataclass(frozen=True)
class EvaluationSummary:
    total_pnl: MeanStd
    gross_pnl_before_fees: MeanStd
    total_fees: MeanStd
    total_reward: MeanStd
    sharpe_ratio: MeanStd
    maximum_drawdown: MeanStd
    max_abs_inventory: MeanStd
    mean_abs_inventory: MeanStd
    final_inventory: MeanStd
    quote_rate: MeanStd
    fill_rate: MeanStd
    total_turnover: MeanStd
    markout_1s: MeanStd
    markout_10s: MeanStd
    profitable_episode_percentage: MeanStd
    median_total_pnl: float
    maximum_observed_inventory: float
    window_ids: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True)
class BaselineParameters:
    spread_bps: float
    queue_fraction: float
    volatility_filter: bool
    imbalance_filter: bool


@dataclass(frozen=True)
class ActorSpec:
    selector: ActionSelector
    quote_spread_bps: float = 5.0
    queue_fraction: float = 0.5


@dataclass(frozen=True)
class SearchCandidate:
    name: str
    learning_rate: float
    ent_coef: float
    gamma: float
    gae_lambda: float
    inventory_penalty_multiplier: float
    size_multipliers: tuple[float, float, float]
    quote_offset_scale: float


@dataclass(frozen=True)
class SearchStage:
    number: int
    keep: int
    timesteps: int
    seeds: tuple[int, ...]
    validation_windows: int


class TemporalCNNExtractor(BaseFeaturesExtractor):
    """Compact temporal encoder for the 30-second residual-policy observation."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 32) -> None:
        super().__init__(observation_space, features_dim)
        feature_count = observation_space.shape[1]
        self.network = nn.Sequential(
            nn.Conv1d(feature_count, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, features_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations.transpose(1, 2)).squeeze(-1)


ActionSelector = Callable[[np.ndarray, float, int], np.ndarray]

BASELINE_SPREAD_CANDIDATES = (2.5, 5.0, 7.5, 10.0, 15.0)
BASELINE_QUEUE_CANDIDATES = (0.25, 0.5, 0.75)
PREVIOUS_FEE_EXPERIMENT = {
    "adaptive_fixed_spread": {"total_pnl": -35.4702, "gross_pnl": -12.7347},
    "adaptive_inventory_skew": {"total_pnl": -38.6765, "gross_pnl": -12.7915},
    "selected_ppo": {"total_pnl": 0.0, "gross_pnl": 0.0},
}
SEARCH_QUEUE_FRACTION = 0.25
SEARCH_CHECKPOINT_STEPS = 50_000
SEARCH_STAGES = (
    SearchStage(1, 3, 60_000, (42,), 8),
    SearchStage(2, 2, 180_000, (42, 100), 18),
    SearchStage(3, 2, 400_000, (42, 100, 200), 36),
)
SEARCH_CANDIDATES = (
    SearchCandidate("base", 3e-4, 0.0, 0.99, 0.95, 1.0, (0.5, 1.0, 2.0), 1.0),
    SearchCandidate("small_sizes", 3e-4, 0.0, 0.99, 0.95, 1.0, (0.25, 0.5, 1.0), 1.0),
    SearchCandidate(
        "high_inventory_control", 3e-4, 0.0, 0.99, 0.95, 2.0, (0.25, 0.5, 1.0), 1.0
    ),
    SearchCandidate(
        "low_inventory_penalty", 3e-4, 0.0, 0.99, 0.95, 0.5, (0.25, 0.5, 1.0), 1.0
    ),
    SearchCandidate("exploration", 3e-4, 0.01, 0.99, 0.95, 1.0, (0.25, 0.5, 1.0), 1.0),
    SearchCandidate(
        "slow_learning", 1e-4, 0.005, 0.995, 0.97, 1.0, (0.25, 0.5, 1.0), 1.0
    ),
    SearchCandidate(
        "wider_quotes", 3e-4, 0.005, 0.99, 0.95, 1.0, (0.25, 0.5, 1.0), 1.5
    ),
    SearchCandidate(
        "wide_low_risk", 1e-4, 0.005, 0.995, 0.97, 2.0, (0.25, 0.5, 1.0), 2.0
    ),
)
RESIDUAL_SCREENING_SEED = 42
RESIDUAL_SCREENING_TIMESTEPS = 250_000
RESIDUAL_FULL_TIMESTEPS = 700_000
RESIDUAL_VARIANTS = ("conservative", "selective_narrow", "asymmetric")
RESIDUAL_FINAL_MODEL = Path("models/real_residual_sac_zero_fee_best.zip")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete real-data experiment.")
    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "evaluate", "all", "full", "optimize", "residual"],
        required=True,
    )
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seeds", default="42,100,200")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--maker-fee", type=float, default=0.0002)
    parser.add_argument("--mandatory-quoting", action="store_true")
    parser.add_argument("--queue-fraction", type=float, default=0.5)
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument("--screening-timesteps", type=int, default=RESIDUAL_SCREENING_TIMESTEPS)
    parser.add_argument("--full-timesteps", type=int, default=RESIDUAL_FULL_TIMESTEPS)
    parser.add_argument("--bc-samples", type=int, default=4_096)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument(
        "--orderbook-dir",
        type=Path,
        default=Path("data/processed/bybit/orderbook/BTCUSDT"),
    )
    parser.add_argument(
        "--trades-dir",
        type=Path,
        default=Path("data/processed/bybit/trades"),
    )
    parser.add_argument(
        "--raw-trades-dir",
        type=Path,
        default=Path("data/raw/bybit/tmp/trades"),
    )
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    if args.mode == "optimize":
        if not np.isclose(args.maker_fee, 0.0) or not args.mandatory_quoting:
            raise ValueError("optimize mode requires --maker-fee 0 --mandatory-quoting")
        if args.max_hours <= 0:
            raise ValueError("--max-hours must be positive")
        args.queue_fraction = SEARCH_QUEUE_FRACTION
    if args.mode == "residual" and not np.isclose(args.maker_fee, 0.0):
        raise ValueError("The first residual SAC experiment requires --maker-fee 0")
    if args.mode == "residual":
        args.queue_fraction = SEARCH_QUEUE_FRACTION
    model_path = resolve_model_path(
        args.model_path,
        maker_fee=args.maker_fee,
        mandatory_quoting=args.mandatory_quoting,
    )
    print(f"maker_fee: {args.maker_fee:.6f}")
    print(f"mandatory_quoting: {str(args.mandatory_quoting).lower()}")
    print(f"queue_fraction: {args.queue_fraction:.2f}")
    if args.mode in {"prepare", "all", "full"}:
        failed = prepare_all_trades(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.trades_dir,
            raw_temp_dir=args.raw_trades_dir,
        )
        print_dataset_status(
            orderbook_dir=args.orderbook_dir,
            trades_dir=args.trades_dir / args.symbol.upper(),
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            failed_dates=failed,
        )
        if failed:
            raise RuntimeError(f"Trade preparation failed for {len(failed)} dates; rerun to resume")
    if args.mode == "prepare":
        return

    trades_symbol_dir = args.trades_dir / args.symbol.upper()
    missing = print_dataset_status(
        orderbook_dir=args.orderbook_dir,
        trades_dir=trades_symbol_dir,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if missing:
        raise RuntimeError("Orderbook/trade coverage is incomplete; run preparation again")
    dates = available_dates(
        orderbook_dir=args.orderbook_dir,
        trades_dir=trades_symbol_dir,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    split = chronological_split(dates)
    print_split(split)

    if args.mode == "optimize":
        run_optimization(
            orderbook_dir=args.orderbook_dir,
            trades_dir=trades_symbol_dir,
            split=split,
            max_hours=args.max_hours,
            symbol=args.symbol,
        )
        return

    if args.mode == "residual":
        run_residual_experiment(
            orderbook_dir=args.orderbook_dir,
            trades_dir=trades_symbol_dir,
            split=split,
            seeds=seeds,
            maker_fee=args.maker_fee,
            screening_timesteps=args.screening_timesteps,
            full_timesteps=args.full_timesteps,
            bc_samples=args.bc_samples,
            force_train=args.force_train,
        )
        return

    if args.mode in {"train", "all", "full"}:
        train_ppo_models(
            orderbook_dir=args.orderbook_dir,
            trades_dir=trades_symbol_dir,
            train_dates=split.train,
            timesteps=args.timesteps,
            seeds=seeds,
            model_path=model_path,
            force_train=args.force_train,
            maker_fee=args.maker_fee,
            mandatory_quoting=args.mandatory_quoting,
            queue_fraction=args.queue_fraction,
        )
    if args.mode in {"evaluate", "all", "full"}:
        run_evaluation(
            orderbook_dir=args.orderbook_dir,
            trades_dir=trades_symbol_dir,
            split=split,
            seeds=seeds,
            model_path=model_path,
            maker_fee=args.maker_fee,
            mandatory_quoting=args.mandatory_quoting,
            queue_fraction=args.queue_fraction,
        )


def resolve_model_path(
    configured_path: Path | None,
    *,
    maker_fee: float,
    mandatory_quoting: bool,
) -> Path:
    if configured_path is not None:
        return configured_path
    if mandatory_quoting and np.isclose(maker_fee, 0.0):
        return Path("models/real_ppo_zero_fee.zip")
    return Path("models/real_ppo.zip")


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def prepare_all_trades(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    raw_temp_dir: Path,
) -> list[str]:
    failed = []
    for current_date in iter_dates(start_date, end_date):
        day = current_date.isoformat()
        try:
            prepare_trade_range(
                symbol=symbol,
                start_date=day,
                end_date=day,
                output_dir=output_dir,
                raw_temp_dir=raw_temp_dir,
                retries=8,
            )
        except Exception as error:  # noqa: BLE001 - continue so one bad day does not stop the year.
            failed.append(day)
            print(f"trades {day}: failed error={error}")
    return failed


def dataset_status(
    *,
    orderbook_dir: Path,
    trades_dir: Path,
    symbol: str,
    start_date: str,
    end_date: str,
) -> tuple[int, int, list[str]]:
    orderbook_count = 0
    trade_count = 0
    missing = []
    symbol = symbol.upper()
    for current_date in iter_dates(start_date, end_date):
        day = current_date.isoformat()
        orderbook_exists = (
            orderbook_dir / f"{symbol}_{day}_orderbook_top10_1s.parquet"
        ).is_file()
        trades_exists = (trades_dir / f"{symbol}_{day}_trades_1s.parquet").is_file()
        orderbook_count += int(orderbook_exists)
        trade_count += int(trades_exists)
        if not orderbook_exists or not trades_exists:
            missing.append(day)
    return orderbook_count, trade_count, missing


def print_dataset_status(
    *,
    orderbook_dir: Path,
    trades_dir: Path,
    symbol: str,
    start_date: str,
    end_date: str,
    failed_dates: list[str] | None = None,
) -> list[str]:
    orderbook_count, trade_count, missing = dataset_status(
        orderbook_dir=orderbook_dir,
        trades_dir=trades_dir,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )
    combined = sorted(set(missing).union(failed_dates or []))
    print(f"orderbook_files: {orderbook_count}")
    print(f"trade_files: {trade_count}")
    print("failed_or_missing_dates: " + (", ".join(combined) if combined else "none"))
    return combined


def available_dates(
    *,
    orderbook_dir: Path,
    trades_dir: Path,
    symbol: str,
    start_date: str,
    end_date: str,
) -> list[date]:
    symbol = symbol.upper()
    result = []
    for current_date in iter_dates(start_date, end_date):
        day = current_date.isoformat()
        orderbook = orderbook_dir / f"{symbol}_{day}_orderbook_top10_1s.parquet"
        trades = trades_dir / f"{symbol}_{day}_trades_1s.parquet"
        if orderbook.is_file() and trades.is_file():
            result.append(current_date)
    if len(result) < 7:
        raise ValueError("At least seven matching orderbook/trade dates are required")
    return result


def chronological_split(dates: list[date]) -> DatasetSplit:
    """Split sorted dates 75/10/15 using largest remainders, never shuffling."""

    ordered = sorted(dates)
    if len(ordered) < 7:
        raise ValueError("At least seven dates are required for train/validation/test")
    weights = [0.75, 0.10, 0.15]
    raw_counts = [len(ordered) * weight for weight in weights]
    counts = [int(value) for value in raw_counts]
    remaining = len(ordered) - sum(counts)
    remainder_order = sorted(
        range(3),
        key=lambda index: raw_counts[index] - counts[index],
        reverse=True,
    )
    for index in remainder_order[:remaining]:
        counts[index] += 1
    train_end = counts[0]
    validation_end = train_end + counts[1]
    return DatasetSplit(
        train=tuple(ordered[:train_end]),
        validation=tuple(ordered[train_end:validation_end]),
        test=tuple(ordered[validation_end:]),
    )


def print_split(split: DatasetSplit) -> None:
    for name in ("train", "validation", "test"):
        values = getattr(split, name)
        print(f"{name}: {values[0]} to {values[-1]} ({len(values)} dates)")


def run_optimization(
    *,
    orderbook_dir: Path,
    trades_dir: Path,
    split: DatasetSplit,
    max_hours: float,
    symbol: str,
    search_dir: Path = Path("models/search_tmp"),
    final_model_path: Path = Path("models/real_ppo_zero_fee_best.zip"),
) -> None:
    search_dir.mkdir(parents=True, exist_ok=True)
    state_path = search_dir / "search_state.json"
    state = load_search_state(state_path)
    if state.get("status") == "complete":
        print(f"optimization already complete: {final_model_path}")
        return
    deadline = time.monotonic() + max_hours * 3_600.0
    candidates_by_name = {candidate.name: candidate for candidate in SEARCH_CANDIDATES}
    active_names = state.get("active_candidates", list(candidates_by_name))

    for stage in SEARCH_STAGES:
        stage_key = str(stage.number)
        if int(state.get("completed_stage", 0)) >= stage.number:
            active_names = state["promotions"][stage_key]
            continue
        windows = build_search_validation_windows(
            orderbook_dir=orderbook_dir,
            symbol=symbol,
            dates=split.validation,
            count=stage.validation_windows,
        )
        stage_results: dict[str, dict[str, object]] = {}
        for candidate_name in active_names:
            candidate = candidates_by_name[candidate_name]
            try:
                seed_summaries = {}
                for seed in stage.seeds:
                    if time.monotonic() >= deadline:
                        pause_search(state, state_path, stage.number)
                        return
                    model_path = train_search_candidate(
                        candidate,
                        seed=seed,
                        target_timesteps=stage.timesteps,
                        orderbook_dir=orderbook_dir,
                        trades_dir=trades_dir,
                        train_dates=split.train,
                        search_dir=search_dir,
                        state=state,
                        state_path=state_path,
                        deadline=deadline,
                    )
                    model = PPO.load(model_path, device="cpu")
                    seed_summaries[seed] = evaluate_actor(
                        ppo_selector(model),
                        orderbook_dir=orderbook_dir,
                        trades_dir=trades_dir,
                        windows=windows,
                        maker_fee=0.0,
                        mandatory_quoting=True,
                        queue_fraction=SEARCH_QUEUE_FRACTION,
                        inventory_penalty_multiplier=(
                            candidate.inventory_penalty_multiplier
                        ),
                        size_multipliers=candidate.size_multipliers,
                        quote_offset_scale=candidate.quote_offset_scale,
                    )
                result = score_search_summaries(seed_summaries)
                result["seeds"] = {
                    str(seed): search_summary_values(summary)
                    for seed, summary in seed_summaries.items()
                }
                stage_results[candidate_name] = result
                print_search_score(stage.number, candidate_name, result)
            except SearchTimeLimitReached:
                pause_search(state, state_path, stage.number)
                return
            except Exception as error:  # noqa: BLE001 - one candidate must not stop the search.
                message = f"{type(error).__name__}: {error}"
                stage_results[candidate_name] = {
                    "eligible": False,
                    "score": None,
                    "error": message,
                }
                state.setdefault("errors", []).append(
                    {"stage": stage.number, "candidate": candidate_name, "error": message}
                )
                print(f"stage={stage.number} candidate={candidate_name} failed: {message}")
            state.setdefault("stage_results", {}).setdefault(stage_key, {}).update(
                {candidate_name: stage_results[candidate_name]}
            )
            save_search_state(state_path, state)
            if time.monotonic() >= deadline:
                pause_search(state, state_path, stage.number)
                return

        promoted = promote_candidates(stage_results, stage.keep)
        if not promoted:
            state["status"] = "failed"
            state["error"] = f"No eligible candidates in stage {stage.number}"
            save_search_state(state_path, state)
            print(state["error"])
            return
        rejected = sorted(set(active_names) - set(promoted))
        cleanup_rejected_models(search_dir, rejected)
        state.setdefault("promotions", {})[stage_key] = promoted
        state["active_candidates"] = promoted
        state["completed_stage"] = stage.number
        state["status"] = "running"
        save_search_state(state_path, state)
        print(f"stage={stage.number} promoted: {', '.join(promoted)}")
        active_names = promoted

    finalize_search_selection(
        state=state,
        state_path=state_path,
        search_dir=search_dir,
        final_model_path=final_model_path,
    )
    if time.monotonic() >= deadline:
        pause_search(state, state_path, SEARCH_STAGES[-1].number)
        return
    run_search_final_test(
        state=state,
        state_path=state_path,
        orderbook_dir=orderbook_dir,
        trades_dir=trades_dir,
        split=split,
        final_model_path=final_model_path,
    )


def initial_search_state() -> dict[str, object]:
    return {
        "version": 1,
        "status": "running",
        "completed_stage": 0,
        "active_candidates": [candidate.name for candidate in SEARCH_CANDIDATES],
        "tasks": {},
        "stage_results": {},
        "promotions": {},
        "errors": [],
    }


def load_search_state(path: Path) -> dict[str, object]:
    if not path.is_file():
        return initial_search_state()
    with path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    if state.get("version") != 1:
        raise ValueError(f"Unsupported search state version: {state.get('version')}")
    if state.get("status") == "paused":
        state["status"] = "running"
    return state


def save_search_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def pause_search(state: dict[str, object], state_path: Path, stage: int) -> None:
    state["status"] = "paused"
    state["paused_stage"] = stage
    save_search_state(state_path, state)
    print(f"max-hours reached after a safe checkpoint; resume stage {stage}")


def search_model_path(search_dir: Path, candidate: str, seed: int) -> Path:
    return search_dir / f"{candidate}_seed{seed}.zip"


def train_search_candidate(
    candidate: SearchCandidate,
    *,
    seed: int,
    target_timesteps: int,
    orderbook_dir: Path,
    trades_dir: Path,
    train_dates: tuple[date, ...],
    search_dir: Path,
    state: dict[str, object],
    state_path: Path,
    deadline: float,
    episode_steps: int = 3_600,
) -> Path:
    task_key = f"{candidate.name}:seed{seed}"
    tasks = state.setdefault("tasks", {})
    task = tasks.setdefault(task_key, {"timesteps": 0, "status": "pending"})
    completed_timesteps = int(task.get("timesteps", 0))
    model_path = search_model_path(search_dir, candidate.name, seed)
    if completed_timesteps >= target_timesteps and model_path.is_file():
        print(
            f"reused candidate={candidate.name} seed={seed} "
            f"timesteps={completed_timesteps}"
        )
        return model_path

    env = RealOrderbookEnv(
        data_dir=orderbook_dir,
        trades_dir=trades_dir,
        start_date=train_dates[0].isoformat(),
        end_date=train_dates[-1].isoformat(),
        episode_steps=episode_steps,
        seed=seed,
        random_start=True,
        maker_fee=0.0,
        mandatory_quoting=True,
        queue_fraction=SEARCH_QUEUE_FRACTION,
        inventory_penalty_multiplier=candidate.inventory_penalty_multiplier,
        size_multipliers=candidate.size_multipliers,
        quote_offset_scale=candidate.quote_offset_scale,
    )
    if completed_timesteps > 0 and model_path.is_file():
        model = PPO.load(model_path, env=env, device="cpu")
    else:
        completed_timesteps = 0
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            n_steps=256,
            batch_size=64,
            learning_rate=candidate.learning_rate,
            ent_coef=candidate.ent_coef,
            gamma=candidate.gamma,
            gae_lambda=candidate.gae_lambda,
            verbose=0,
            device="cpu",
        )
    while completed_timesteps < target_timesteps:
        chunk = min(SEARCH_CHECKPOINT_STEPS, target_timesteps - completed_timesteps)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        completed_timesteps += chunk
        atomic_save_model(model, model_path)
        task.update(
            {
                "timesteps": completed_timesteps,
                "status": (
                    "complete" if completed_timesteps >= target_timesteps else "checkpointed"
                ),
                "model_path": str(model_path),
            }
        )
        save_search_state(state_path, state)
        print(
            f"checkpoint candidate={candidate.name} seed={seed} "
            f"timesteps={completed_timesteps}"
        )
        if time.monotonic() >= deadline and completed_timesteps < target_timesteps:
            env.close()
            raise SearchTimeLimitReached
    env.close()
    return model_path


class SearchTimeLimitReached(Exception):
    """Signal that a candidate stopped at an atomic checkpoint."""


def atomic_save_model(model: PPO | SAC, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp.zip")
    temporary.unlink(missing_ok=True)
    model.save(temporary)
    os.replace(temporary, path)


def build_search_validation_windows(
    *,
    orderbook_dir: Path,
    symbol: str,
    dates: tuple[date, ...],
    count: int,
) -> list[EvaluationWindow]:
    if not 1 <= count <= len(dates):
        raise ValueError("validation window count must fit inside validation dates")
    indices = np.linspace(0, len(dates) - 1, num=count, dtype=int)
    selected_dates = tuple(dates[int(index)] for index in indices)
    return build_evaluation_windows(
        orderbook_dir=orderbook_dir,
        symbol=symbol,
        dates=selected_dates,
        seeds=[42],
        minimum_episodes=count,
    )


def search_summary_values(summary: EvaluationSummary) -> dict[str, float]:
    return {
        "mean_validation_pnl": summary.total_pnl.mean,
        "std_validation_pnl": summary.total_pnl.std,
        "mean_max_drawdown": summary.maximum_drawdown.mean,
        "mean_fill_rate": summary.fill_rate.mean,
        "mean_quote_rate": summary.quote_rate.mean,
        "maximum_observed_inventory": summary.maximum_observed_inventory,
    }


def score_search_summaries(
    summaries: dict[int, EvaluationSummary],
) -> dict[str, object]:
    values = list(summaries.values())
    means = np.asarray([summary.total_pnl.mean for summary in values], dtype=float)
    pooled_mean = float(np.mean(means))
    pooled_variance = float(
        np.mean(
            [
                summary.total_pnl.std**2 + (summary.total_pnl.mean - pooled_mean) ** 2
                for summary in values
            ]
        )
    )
    pooled_std = float(np.sqrt(pooled_variance))
    mean_drawdown = float(np.mean([summary.maximum_drawdown.mean for summary in values]))
    mean_fill_rate = float(np.mean([summary.fill_rate.mean for summary in values]))
    mean_quote_rate = float(np.mean([summary.quote_rate.mean for summary in values]))
    maximum_inventory = max(summary.maximum_observed_inventory for summary in values)
    numeric = [
        pooled_mean,
        pooled_std,
        mean_drawdown,
        mean_fill_rate,
        mean_quote_rate,
        maximum_inventory,
    ]
    eligible = bool(
        all(np.isfinite(value) for value in numeric)
        and mean_fill_rate >= 0.001
        and mean_quote_rate >= 0.999
        and maximum_inventory <= 0.02 + 1e-12
    )
    score = pooled_mean - 0.25 * pooled_std - 0.10 * mean_drawdown
    return {
        "eligible": eligible,
        "score": score if eligible else None,
        "mean_validation_pnl": pooled_mean,
        "std_validation_pnl": pooled_std,
        "mean_max_drawdown": mean_drawdown,
        "mean_fill_rate": mean_fill_rate,
        "mean_quote_rate": mean_quote_rate,
        "maximum_observed_inventory": maximum_inventory,
    }


def promote_candidates(
    stage_results: dict[str, dict[str, object]],
    keep: int,
) -> list[str]:
    eligible = [
        (name, result)
        for name, result in stage_results.items()
        if result.get("eligible") and result.get("score") is not None
    ]
    eligible.sort(key=lambda item: (-float(item[1]["score"]), item[0]))
    return [name for name, _ in eligible[:keep]]


def print_search_score(stage: int, candidate: str, result: dict[str, object]) -> None:
    print(
        f"stage={stage} candidate={candidate} eligible={result['eligible']} "
        f"mean_validation_pnl={result['mean_validation_pnl']:.4f} "
        f"std_validation_pnl={result['std_validation_pnl']:.4f} "
        f"mean_max_drawdown={result['mean_max_drawdown']:.4f} "
        f"score={result['score']}"
    )


def cleanup_rejected_models(search_dir: Path, candidate_names: list[str]) -> None:
    for candidate_name in candidate_names:
        for seed in (42, 100, 200):
            search_model_path(search_dir, candidate_name, seed).unlink(missing_ok=True)


def finalize_search_selection(
    *,
    state: dict[str, object],
    state_path: Path,
    search_dir: Path,
    final_model_path: Path,
) -> None:
    if state.get("selection_complete"):
        return
    stage_results = state["stage_results"][str(SEARCH_STAGES[-1].number)]
    best_configuration = promote_candidates(stage_results, 1)[0]
    seed_results = stage_results[best_configuration]["seeds"]
    best_seed = max(
        (int(seed) for seed in seed_results),
        key=lambda seed: (
            search_result_score(seed_results[str(seed)]),
            -seed,
        ),
    )
    source = search_model_path(search_dir, best_configuration, best_seed)
    atomic_copy_model(source, final_model_path)
    state["selected_configuration"] = best_configuration
    state["selected_seed"] = best_seed
    state["selection_complete"] = True
    state["best_model_path"] = str(final_model_path)
    save_search_state(state_path, state)
    print(f"selected_configuration: {best_configuration}")
    print(f"selected_seed: {best_seed}")
    print(f"best_model: {final_model_path}")


def search_result_score(values: dict[str, float]) -> float:
    return (
        float(values["mean_validation_pnl"])
        - 0.25 * float(values["std_validation_pnl"])
        - 0.10 * float(values["mean_max_drawdown"])
    )


def atomic_copy_model(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.stem}.tmp.zip")
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def residual_model_path(
    variant: str,
    seed: int,
    timesteps: int,
    *,
    screening: bool,
) -> Path:
    suffix = "" if screening else f"_{timesteps}"
    return Path(f"models/search_tmp/residual_sac_{variant}_seed{seed}{suffix}.zip")


def residual_selector(model: SAC) -> ActionSelector:
    def select(observation: np.ndarray, inventory: float, seed: int) -> np.ndarray:
        del inventory, seed
        return np.asarray(model.predict(observation, deterministic=True)[0], dtype=np.float32)

    return select


def collect_behavior_cloning_batch(
    *,
    orderbook_dir: Path,
    trades_dir: Path,
    train_dates: tuple[date, ...],
    base_parameters: BaselineParameters,
    sample_count: int,
    seed: int,
    residual_variant: str = "legacy",
) -> tuple[np.ndarray, np.ndarray, tuple[date, ...]]:
    """Generate baseline expert examples directly from train-day replay."""
    if not train_dates:
        raise ValueError("Behaviour cloning requires train dates")
    rng = np.random.default_rng(seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    used_dates: list[date] = []
    dates = list(train_dates)
    while len(observations) < sample_count:
        current_date = dates[int(rng.integers(len(dates)))]
        day = current_date.isoformat()
        env = RealOrderbookEnv(
            data_dir=orderbook_dir,
            trades_dir=trades_dir,
            start_date=day,
            end_date=day,
            episode_steps=min(600, sample_count - len(observations)),
            seed=seed + len(used_dates),
            random_start=True,
            maker_fee=0.0,
            queue_fraction=base_parameters.queue_fraction,
            quote_spread_bps=base_parameters.spread_bps,
            residual_continuous=True,
            residual_variant=residual_variant,
            base_spread_bps=base_parameters.spread_bps,
            base_volatility_filter=base_parameters.volatility_filter,
            base_imbalance_filter=base_parameters.imbalance_filter,
        )
        observation, _ = env.reset()
        used_dates.append(current_date)
        truncated = False
        while not truncated and len(observations) < sample_count:
            expert_action = env.neutral_residual_action(residual_variant)
            observations.append(observation.copy())
            actions.append(expert_action.copy())
            observation, _, _, truncated, _ = env.step(expert_action)
        env.close()
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        tuple(used_dates),
    )


def pretrain_residual_actor(
    model: SAC,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    epochs: int = 5,
    batch_size: int = 256,
) -> tuple[float, float]:
    """Warm-start the SAC actor against neutral residual baseline actions."""
    observation_tensor = torch.as_tensor(observations, device=model.device)
    action_tensor = torch.as_tensor(actions, device=model.device)

    def imitation_error() -> float:
        with torch.no_grad():
            predicted = model.actor(observation_tensor, deterministic=True)
            return float(torch.mean((predicted - action_tensor) ** 2).cpu())

    before = imitation_error()
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        for start in range(0, len(observations), batch_size):
            indices = rng.permutation(len(observations))[start : start + batch_size]
            batch_observations = observation_tensor[indices]
            batch_actions = action_tensor[indices]
            predicted = model.actor(batch_observations, deterministic=True)
            loss = torch.mean((predicted - batch_actions) ** 2)
            model.actor.optimizer.zero_grad()
            loss.backward()
            model.actor.optimizer.step()
    return before, imitation_error()


def train_residual_sac(
    *,
    orderbook_dir: Path,
    trades_dir: Path,
    train_dates: tuple[date, ...],
    base_parameters: BaselineParameters,
    seed: int,
    residual_variant: str,
    target_timesteps: int,
    bc_samples: int,
    screening: bool = False,
    force_train: bool = False,
    warm_start_path: Path | None = None,
    warm_start_timesteps: int = 0,
) -> tuple[Path, tuple[float, float] | None]:
    path = residual_model_path(
        residual_variant,
        seed,
        target_timesteps,
        screening=screening,
    )
    if path.is_file() and not force_train:
        print(f"reused residual SAC seed={seed} timesteps={target_timesteps} model={path}")
        return path, None
    env = RealOrderbookEnv(
        data_dir=orderbook_dir,
        trades_dir=trades_dir,
        start_date=train_dates[0].isoformat(),
        end_date=train_dates[-1].isoformat(),
        episode_steps=3_600,
        seed=seed,
        random_start=True,
        maker_fee=0.0,
        queue_fraction=base_parameters.queue_fraction,
        quote_spread_bps=base_parameters.spread_bps,
        residual_continuous=True,
        residual_variant=residual_variant,
        base_spread_bps=base_parameters.spread_bps,
        base_volatility_filter=base_parameters.volatility_filter,
        base_imbalance_filter=base_parameters.imbalance_filter,
    )
    learned_timesteps = 0
    imitation = None
    if warm_start_path is not None and warm_start_path.is_file():
        model = SAC.load(warm_start_path, env=env, device="cpu")
        learned_timesteps = warm_start_timesteps
    else:
        model = SAC(
            "MlpPolicy",
            env,
            seed=seed,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=256,
            train_freq=4,
            gradient_steps=1,
            policy_kwargs={
                "features_extractor_class": TemporalCNNExtractor,
                "features_extractor_kwargs": {"features_dim": 32},
                "net_arch": [64, 64],
            },
            verbose=0,
            device="cpu",
        )
        examples, expert_actions, used_dates = collect_behavior_cloning_batch(
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            train_dates=train_dates,
            base_parameters=base_parameters,
            sample_count=bc_samples,
            seed=seed,
            residual_variant=residual_variant,
        )
        if not set(used_dates).issubset(train_dates):
            raise RuntimeError("Behaviour cloning accessed a non-train date")
        imitation = pretrain_residual_actor(model, examples, expert_actions)
        print(
            f"baseline_imitation_error variant={residual_variant} seed={seed}: "
            f"before={imitation[0]:.6f} after={imitation[1]:.6f}"
        )
    remaining = target_timesteps - learned_timesteps
    if remaining > 0:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False)
    atomic_save_model(model, path)
    env.close()
    print(
        f"trained residual SAC variant={residual_variant} seed={seed} "
        f"timesteps={target_timesteps} model={path}"
    )
    return path, imitation


def residual_validation_result(
    summary: EvaluationSummary,
    *,
    baseline_summary: EvaluationSummary,
) -> dict[str, float | bool | None]:
    components = validation_score_components(summary)
    baseline_components = validation_score_components(baseline_summary)
    quote_threshold = 0.5 * baseline_summary.quote_rate.mean
    fill_threshold = 0.5 * baseline_summary.fill_rate.mean
    quote_active = summary.quote_rate.mean >= quote_threshold
    fill_active = summary.fill_rate.mean >= fill_threshold
    numeric_values = (
        components["mean_validation_pnl"],
        components["std_validation_pnl"],
        components["mean_max_drawdown"],
        components["score"],
        summary.quote_rate.mean,
        summary.fill_rate.mean,
        summary.maximum_observed_inventory,
    )
    valid_metrics = all(np.isfinite(value) for value in numeric_values)
    inventory_safe = bool(summary.maximum_observed_inventory <= 0.02 + 1e-12)
    eligible = quote_active and fill_active and inventory_safe and valid_metrics
    positive_pnl = summary.total_pnl.mean > 0.0
    beats_baseline_score = components["score"] > baseline_components["score"]
    advances = eligible and positive_pnl and beats_baseline_score
    return {
        "eligible": eligible,
        "advances": advances,
        "valid_metrics": valid_metrics,
        "inventory_safe": inventory_safe,
        "quote_active": quote_active,
        "fill_active": fill_active,
        "quote_threshold": quote_threshold,
        "fill_threshold": fill_threshold,
        "positive_mean_pnl": positive_pnl,
        "beats_baseline_score": beats_baseline_score,
        "baseline_score": baseline_components["score"],
        **components,
    }


def validation_score_components(summary: EvaluationSummary) -> dict[str, float]:
    mean_pnl = summary.total_pnl.mean
    std_pnl = summary.total_pnl.std
    mean_drawdown = summary.maximum_drawdown.mean
    return {
        "mean_validation_pnl": mean_pnl,
        "std_validation_pnl": std_pnl,
        "mean_max_drawdown": mean_drawdown,
        "score": mean_pnl - 0.25 * std_pnl - 0.10 * mean_drawdown,
    }


def print_validation_score(
    name: str,
    summary: EvaluationSummary,
    baseline_summary: EvaluationSummary,
) -> None:
    result = residual_validation_result(summary, baseline_summary=baseline_summary)
    print(
        f"validation_score candidate={name} "
        f"mean_pnl={result['mean_validation_pnl']:.6f} "
        f"pnl_std={result['std_validation_pnl']:.6f} "
        f"mean_max_drawdown={result['mean_max_drawdown']:.6f} "
        f"score={result['score']:.6f} baseline_score={result['baseline_score']:.6f} "
        f"quote_rate={summary.quote_rate.mean:.6f} "
        f"quote_threshold={result['quote_threshold']:.6f} "
        f"fill_rate={summary.fill_rate.mean:.6f} "
        f"fill_threshold={result['fill_threshold']:.6f} "
        f"valid_metrics={result['valid_metrics']} "
        f"inventory_safe={result['inventory_safe']} "
        f"eligible={result['eligible']} advances={result['advances']}"
    )


def select_residual_variant(
    validation_summaries: dict[str, EvaluationSummary],
    *,
    baseline_summary: EvaluationSummary,
) -> str | None:
    advancing = []
    for variant, summary in validation_summaries.items():
        result = residual_validation_result(summary, baseline_summary=baseline_summary)
        if result["advances"]:
            advancing.append((float(result["score"]), variant))
    if not advancing:
        return None
    return max(advancing, key=lambda item: (item[0], item[1]))[1]


def select_residual_seed(
    validation_summaries: dict[int, EvaluationSummary],
    *,
    baseline_summary: EvaluationSummary,
) -> int | None:
    eligible = []
    for seed, summary in validation_summaries.items():
        result = residual_validation_result(summary, baseline_summary=baseline_summary)
        if result["advances"]:
            eligible.append((float(result["score"]), seed))
    if not eligible:
        return None
    return max(eligible, key=lambda item: (item[0], -item[1]))[1]


def previous_ppo_settings() -> tuple[PPO, SearchCandidate]:
    state_path = Path("models/search_tmp/search_state.json")
    model_path = Path("models/real_ppo_zero_fee_best.zip")
    if not state_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("Run make optimize-real-zero-fee before residual SAC")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    candidate = next(
        item for item in SEARCH_CANDIDATES if item.name == state["selected_configuration"]
    )
    return PPO.load(model_path, device="cpu"), candidate


def run_residual_experiment(
    *,
    orderbook_dir: Path,
    trades_dir: Path,
    split: DatasetSplit,
    seeds: list[int],
    maker_fee: float,
    screening_timesteps: int,
    full_timesteps: int,
    bc_samples: int,
    force_train: bool,
) -> None:
    if not np.isclose(maker_fee, 0.0):
        raise ValueError("Residual SAC screening is currently defined for zero maker fees")
    base_parameters = BaselineParameters(15.0, SEARCH_QUEUE_FRACTION, False, True)
    print(f"residual_base_policy: {format_baseline_parameters(base_parameters)}")
    validation_windows = build_evaluation_windows(
        orderbook_dir=orderbook_dir,
        symbol="BTCUSDT",
        dates=split.validation,
        seeds=[42],
        minimum_episodes=len(split.validation),
    )
    inventory_actor = baseline_actor("inventory", base_parameters, False)
    baseline_summary = evaluate_actor(
        inventory_actor.selector,
        orderbook_dir=orderbook_dir,
        trades_dir=trades_dir,
        windows=validation_windows,
        maker_fee=0.0,
        queue_fraction=base_parameters.queue_fraction,
        quote_spread_bps=base_parameters.spread_bps,
    )
    baseline_components = validation_score_components(baseline_summary)
    print(
        "baseline_score_components: "
        f"mean_pnl={baseline_components['mean_validation_pnl']:.6f} "
        f"pnl_std={baseline_components['std_validation_pnl']:.6f} "
        f"mean_max_drawdown={baseline_components['mean_max_drawdown']:.6f} "
        f"score={baseline_components['score']:.6f}"
    )

    screening_paths: dict[str, Path] = {}
    screening_summaries: dict[str, EvaluationSummary] = {}
    for variant in RESIDUAL_VARIANTS:
        path, _ = train_residual_sac(
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            train_dates=split.train,
            base_parameters=base_parameters,
            seed=RESIDUAL_SCREENING_SEED,
            residual_variant=variant,
            target_timesteps=screening_timesteps,
            bc_samples=bc_samples,
            screening=True,
            force_train=force_train,
        )
        screening_paths[variant] = path
        screening_summaries[variant] = evaluate_residual_model(
            path,
            variant=variant,
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            windows=validation_windows,
            base_parameters=base_parameters,
        )

    print("screening validation metrics:")
    print_metrics_table({"baseline": baseline_summary, **screening_summaries})
    for variant in RESIDUAL_VARIANTS:
        print_validation_score(variant, screening_summaries[variant], baseline_summary)
    selected_variant = select_residual_variant(
        screening_summaries,
        baseline_summary=baseline_summary,
    )
    print(f"selected_residual_variant: {selected_variant or 'none'}")
    if selected_variant is None:
        print("full_training_triggered: false")
        return

    print("full_training_triggered: true")
    validation_summaries: dict[int, EvaluationSummary] = {}
    full_paths: dict[int, Path] = {}
    for seed in seeds:
        warm_start = (
            screening_paths[selected_variant]
            if seed == RESIDUAL_SCREENING_SEED
            else None
        )
        path, _ = train_residual_sac(
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            train_dates=split.train,
            base_parameters=base_parameters,
            seed=seed,
            residual_variant=selected_variant,
            target_timesteps=full_timesteps,
            bc_samples=bc_samples,
            force_train=force_train,
            warm_start_path=warm_start,
            warm_start_timesteps=(
                screening_timesteps if seed == RESIDUAL_SCREENING_SEED else 0
            ),
        )
        full_paths[seed] = path
        validation_summaries[seed] = evaluate_residual_model(
            path,
            variant=selected_variant,
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            windows=validation_windows,
            base_parameters=base_parameters,
        )
        print_validation_score(
            f"{selected_variant}_seed{seed}",
            validation_summaries[seed],
            baseline_summary,
        )
    selected_seed = select_residual_seed(
        validation_summaries,
        baseline_summary=baseline_summary,
    )
    if selected_seed is None:
        print("multi_seed_selection: none")
        print("test_evaluation_triggered: false")
        return
    atomic_copy_model(full_paths[selected_seed], RESIDUAL_FINAL_MODEL)
    print(f"selected_residual_seed: {selected_seed}")
    print(f"best_residual_model: {RESIDUAL_FINAL_MODEL}")

    test_windows = build_residual_test_windows(
        orderbook_dir=orderbook_dir,
        split=split,
        promotion_complete=True,
    )
    previous_model, previous_candidate = previous_ppo_settings()
    fixed_actor = baseline_actor("fixed", base_parameters, False)
    residual_model = SAC.load(RESIDUAL_FINAL_MODEL, device="cpu")
    results = {
        "adaptive_fixed_spread": evaluate_actor(
            fixed_actor.selector,
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            windows=test_windows,
            maker_fee=0.0,
            queue_fraction=base_parameters.queue_fraction,
            quote_spread_bps=base_parameters.spread_bps,
        ),
        "adaptive_inventory_skew": evaluate_actor(
            inventory_actor.selector,
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            windows=test_windows,
            maker_fee=0.0,
            queue_fraction=base_parameters.queue_fraction,
            quote_spread_bps=base_parameters.spread_bps,
        ),
        "previous_best_ppo": evaluate_actor(
            ppo_selector(previous_model),
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            windows=test_windows,
            maker_fee=0.0,
            mandatory_quoting=True,
            queue_fraction=SEARCH_QUEUE_FRACTION,
            inventory_penalty_multiplier=previous_candidate.inventory_penalty_multiplier,
            size_multipliers=previous_candidate.size_multipliers,
            quote_offset_scale=previous_candidate.quote_offset_scale,
        ),
        "residual_sac": evaluate_actor(
            residual_selector(residual_model),
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            windows=test_windows,
            maker_fee=0.0,
            queue_fraction=base_parameters.queue_fraction,
            quote_spread_bps=base_parameters.spread_bps,
            residual_continuous=True,
            residual_variant=selected_variant,
            base_parameters=base_parameters,
        ),
    }
    print("final residual SAC test comparison:")
    print_metrics_table(results)
    for name, summary in results.items():
        print(
            f"  {name}: median_pnl={summary.median_total_pnl:.4f} "
            f"total_pnl={summary.total_pnl.mean * len(test_windows):.4f}"
        )


def evaluate_residual_model(
    path: Path,
    *,
    variant: str,
    orderbook_dir: Path,
    trades_dir: Path,
    windows: list[EvaluationWindow],
    base_parameters: BaselineParameters,
) -> EvaluationSummary:
    model = SAC.load(path, device="cpu")
    return evaluate_actor(
        residual_selector(model),
        orderbook_dir=orderbook_dir,
        trades_dir=trades_dir,
        windows=windows,
        maker_fee=0.0,
        queue_fraction=base_parameters.queue_fraction,
        quote_spread_bps=base_parameters.spread_bps,
        residual_continuous=True,
        residual_variant=variant,
        base_parameters=base_parameters,
    )


def build_residual_test_windows(
    *,
    orderbook_dir: Path,
    split: DatasetSplit,
    promotion_complete: bool,
) -> list[EvaluationWindow]:
    if not promotion_complete:
        raise RuntimeError("Test data cannot be accessed before residual promotion")
    return build_evaluation_windows(
        orderbook_dir=orderbook_dir,
        symbol="BTCUSDT",
        dates=split.test,
        seeds=[42],
        minimum_episodes=len(split.test),
    )


def run_search_final_test(
    *,
    state: dict[str, object],
    state_path: Path,
    orderbook_dir: Path,
    trades_dir: Path,
    split: DatasetSplit,
    final_model_path: Path,
) -> None:
    if not state.get("selection_complete"):
        raise RuntimeError("Test data cannot be accessed before final selection")
    if state.get("test_complete"):
        return
    candidate = next(
        item for item in SEARCH_CANDIDATES if item.name == state["selected_configuration"]
    )
    model = PPO.load(final_model_path, device="cpu")
    windows = build_evaluation_windows(
        orderbook_dir=orderbook_dir,
        symbol="BTCUSDT",
        dates=split.test,
        seeds=[42],
    )
    baseline_parameters = BaselineParameters(15.0, 0.25, False, True)
    actors = {
        "adaptive_fixed_spread": baseline_actor("fixed", baseline_parameters, True),
        "adaptive_inventory_skew": baseline_actor(
            "inventory", baseline_parameters, True
        ),
        "selected_ppo": ActorSpec(
            ppo_selector(model),
            queue_fraction=SEARCH_QUEUE_FRACTION,
        ),
    }
    results = {}
    for name, actor in actors.items():
        candidate_settings = candidate if name == "selected_ppo" else None
        results[name] = evaluate_actor(
            actor.selector,
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            windows=windows,
            maker_fee=0.0,
            mandatory_quoting=True,
            queue_fraction=actor.queue_fraction,
            quote_spread_bps=actor.quote_spread_bps,
            inventory_penalty_multiplier=(
                candidate_settings.inventory_penalty_multiplier
                if candidate_settings
                else 1.0
            ),
            size_multipliers=(
                candidate_settings.size_multipliers
                if candidate_settings
                else (0.5, 1.0, 2.0)
            ),
            quote_offset_scale=(
                candidate_settings.quote_offset_scale if candidate_settings else 1.0
            ),
        )
    print("final search test results:")
    print_metrics_table(results)
    print("median total_pnl:")
    for name, summary in results.items():
        print(f"  {name}: {summary.median_total_pnl:.4f}")
    selected = results["selected_ppo"]
    print(
        "previous PPO seed42 comparison: "
        f"previous_total_pnl=-0.6787 current_total_pnl={selected.total_pnl.mean:.4f}"
    )
    state["test_complete"] = True
    state["status"] = "complete"
    state["test_results"] = {
        name: {
            "mean_total_pnl": summary.total_pnl.mean,
            "std_total_pnl": summary.total_pnl.std,
            "median_total_pnl": summary.median_total_pnl,
            "profitable_episode_percentage": (
                summary.profitable_episode_percentage.mean
            ),
        }
        for name, summary in results.items()
    }
    save_search_state(state_path, state)


def train_ppo_models(
    *,
    orderbook_dir: Path,
    trades_dir: Path,
    train_dates: tuple[date, ...],
    timesteps: int,
    seeds: list[int],
    model_path: Path,
    episode_steps: int = 3_600,
    force_train: bool = False,
    maker_fee: float = 0.0002,
    mandatory_quoting: bool = False,
    queue_fraction: float = 0.5,
) -> dict[int, Path]:
    if timesteps < 1:
        raise ValueError("timesteps must be positive")
    paths = {}
    for seed in seeds:
        path = model_path_for_seed(model_path, seed)
        if path.is_file() and not force_train:
            print(f"reused PPO seed={seed} model={path}")
            paths[seed] = path
            continue
        env = RealOrderbookEnv(
            data_dir=orderbook_dir,
            trades_dir=trades_dir,
            start_date=train_dates[0].isoformat(),
            end_date=train_dates[-1].isoformat(),
            episode_steps=episode_steps,
            seed=seed,
            random_start=True,
            maker_fee=maker_fee,
            mandatory_quoting=mandatory_quoting,
            queue_fraction=queue_fraction,
        )
        n_steps = min(256, episode_steps)
        batch_size = min(64, n_steps)
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            n_steps=n_steps,
            batch_size=batch_size,
            verbose=0,
            device="cpu",
        )
        model.learn(total_timesteps=timesteps)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f"{path.stem}.tmp.zip")
        temporary_path.unlink(missing_ok=True)
        model.save(temporary_path)
        os.replace(temporary_path, path)
        env.close()
        paths[seed] = path
        print(
            f"trained PPO seed={seed} timesteps={timesteps} maker_fee={maker_fee:.6f} "
            f"mandatory_quoting={str(mandatory_quoting).lower()} model={path}"
        )
    return paths


def model_path_for_seed(base_path: Path, seed: int) -> Path:
    return base_path.with_name(f"{base_path.stem}_seed{seed}.zip")


def build_evaluation_windows(
    *,
    orderbook_dir: Path,
    symbol: str,
    dates: tuple[date, ...],
    seeds: list[int],
    episode_steps: int = 3_600,
    minimum_episodes: int = 20,
) -> list[EvaluationWindow]:
    base = [(seed, current_date) for seed in seeds for current_date in dates]
    target_count = max(minimum_episodes, len(base))
    windows = []
    for index in range(target_count):
        seed, current_date = base[index % len(base)]
        repetition = index // len(base)
        path = orderbook_dir / (
            f"{symbol.upper()}_{current_date.isoformat()}_orderbook_top10_1s.parquet"
        )
        available_steps = pq.ParquetFile(path).metadata.num_rows - 1
        steps = min(episode_steps, available_steps)
        max_start = available_steps - steps
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, current_date.toordinal(), repetition])
        )
        start_index = int(rng.integers(max_start + 1)) if max_start else 0
        windows.append(EvaluationWindow(current_date, start_index, steps, seed))
    return windows


def run_evaluation(
    *,
    orderbook_dir: Path,
    trades_dir: Path,
    split: DatasetSplit,
    seeds: list[int],
    model_path: Path,
    maker_fee: float = 0.0002,
    mandatory_quoting: bool = False,
    queue_fraction: float = 0.5,
) -> tuple[int, dict[str, EvaluationSummary], dict[str, BaselineParameters]]:
    models = load_models(model_path, seeds)
    validation_windows = build_evaluation_windows(
        orderbook_dir=orderbook_dir,
        symbol="BTCUSDT",
        dates=split.validation,
        seeds=[42],
    )
    validation_results = {
        seed: evaluate_actor(
            ppo_selector(model),
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            windows=validation_windows,
            maker_fee=maker_fee,
            mandatory_quoting=mandatory_quoting,
            queue_fraction=queue_fraction,
        )
        for seed, model in models.items()
    }
    validation_rewards = {
        seed: summary.total_reward.mean for seed, summary in validation_results.items()
    }
    selected_seed = select_best_seed(validation_rewards)
    print("validation PPO results:")
    for seed in seeds:
        reward = validation_results[seed].total_reward
        print(f"  seed={seed} total_reward={reward.mean:.4f} +/- {reward.std:.4f}")
    print(f"selected_best_seed: {selected_seed}")

    selected_baselines = {
        "adaptive_fixed_spread": tune_baseline(
            "fixed",
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            windows=validation_windows,
            maker_fee=maker_fee,
            mandatory_quoting=mandatory_quoting,
        ),
        "adaptive_inventory_skew": tune_baseline(
            "inventory",
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            windows=validation_windows,
            maker_fee=maker_fee,
            mandatory_quoting=mandatory_quoting,
        ),
    }
    print("selected baseline parameters (validation only):")
    for name, parameters in selected_baselines.items():
        print(f"  {name}: {format_baseline_parameters(parameters)}")

    test_windows = build_evaluation_windows(
        orderbook_dir=orderbook_dir,
        symbol="BTCUSDT",
        dates=split.test,
        seeds=[42],
    )
    actors: dict[str, ActorSpec] = {
        "adaptive_fixed_spread": baseline_actor(
            "fixed", selected_baselines["adaptive_fixed_spread"], mandatory_quoting
        ),
        "adaptive_inventory_skew": baseline_actor(
            "inventory", selected_baselines["adaptive_inventory_skew"], mandatory_quoting
        ),
    }
    actors.update(
        {
            f"ppo_seed{seed}": ActorSpec(
                ppo_selector(models[seed]),
                queue_fraction=queue_fraction,
            )
            for seed in seeds
        }
    )
    results = {
        name: evaluate_actor(
            actor.selector,
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            windows=test_windows,
            maker_fee=maker_fee,
            mandatory_quoting=mandatory_quoting,
            queue_fraction=actor.queue_fraction,
            quote_spread_bps=actor.quote_spread_bps,
        )
        for name, actor in actors.items()
    }
    results[f"selected_ppo_seed{selected_seed}"] = results[f"ppo_seed{selected_seed}"]
    print(f"maker_fee: {maker_fee:.6f}")
    print(f"test_episodes: {len(test_windows)}")
    print_metrics_table(results)
    print_profitability(results)
    print_previous_comparison(results, selected_seed)
    return selected_seed, results, selected_baselines


def baseline_candidates() -> list[BaselineParameters]:
    return [
        BaselineParameters(spread, queue, volatility_filter, imbalance_filter)
        for spread in BASELINE_SPREAD_CANDIDATES
        for queue in BASELINE_QUEUE_CANDIDATES
        for volatility_filter in (False, True)
        for imbalance_filter in (False, True)
    ]


def tune_baseline(
    kind: str,
    *,
    orderbook_dir: Path,
    trades_dir: Path,
    windows: list[EvaluationWindow],
    maker_fee: float,
    mandatory_quoting: bool,
) -> BaselineParameters:
    validation_rewards = {}
    for parameters in baseline_candidates():
        actor = baseline_actor(kind, parameters, mandatory_quoting)
        summary = evaluate_actor(
            actor.selector,
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            windows=windows,
            maker_fee=maker_fee,
            mandatory_quoting=mandatory_quoting,
            queue_fraction=actor.queue_fraction,
            quote_spread_bps=actor.quote_spread_bps,
        )
        validation_rewards[parameters] = summary.total_reward.mean
    return select_best_baseline(validation_rewards)


def select_best_baseline(
    validation_rewards: dict[BaselineParameters, float],
) -> BaselineParameters:
    if not validation_rewards:
        raise ValueError("Validation rewards are required for baseline selection")
    return max(
        validation_rewards,
        key=lambda parameters: (
            validation_rewards[parameters],
            -parameters.spread_bps,
            -parameters.queue_fraction,
            not parameters.volatility_filter,
            not parameters.imbalance_filter,
        ),
    )


def baseline_actor(
    kind: str,
    parameters: BaselineParameters,
    mandatory_quoting: bool,
) -> ActorSpec:
    common = {
        "spread_bps": parameters.spread_bps,
        "volatility_filter": parameters.volatility_filter,
        "imbalance_filter": parameters.imbalance_filter,
    }
    if kind == "fixed":
        strategy: BaseStrategy = FixedSpreadStrategy(**common)
    elif kind == "inventory":
        strategy = InventorySkewStrategy(threshold=0.25, **common)
    else:
        raise ValueError(f"Unknown baseline kind: {kind}")
    return ActorSpec(
        baseline_selector(strategy, mandatory_quoting=mandatory_quoting),
        quote_spread_bps=parameters.spread_bps,
        queue_fraction=parameters.queue_fraction,
    )


def format_baseline_parameters(parameters: BaselineParameters) -> str:
    return (
        f"spread_bps={parameters.spread_bps:g}, "
        f"queue_fraction={parameters.queue_fraction:.2f}, "
        f"volatility_filter={str(parameters.volatility_filter).lower()}, "
        f"imbalance_filter={str(parameters.imbalance_filter).lower()}"
    )


def load_models(model_path: Path, seeds: list[int]) -> dict[int, PPO]:
    models = {}
    for seed in seeds:
        path = model_path_for_seed(model_path, seed)
        if not path.is_file():
            raise FileNotFoundError(f"Missing PPO model for seed {seed}: {path}")
        models[seed] = PPO.load(path, device="cpu")
    return models


def select_best_seed(validation_rewards: dict[int, float]) -> int:
    if not validation_rewards:
        raise ValueError("Validation rewards are required for seed selection")
    return max(validation_rewards, key=lambda seed: (validation_rewards[seed], -seed))


def baseline_selector(
    strategy: BaseStrategy,
    *,
    mandatory_quoting: bool = False,
) -> ActionSelector:
    def select(observation: np.ndarray, inventory: float, seed: int) -> np.ndarray:
        del seed
        quote_action = strategy.select_action(
            {
                "inventory": inventory,
                "inventory_ratio": float(observation[0]),
                "volatility": float(observation[4]),
                "orderbook_imbalance": float(observation[2]),
                "trade_imbalance": float(observation[5]),
            }
        )
        policy_action = quote_action - 1 if mandatory_quoting else quote_action
        return np.array([policy_action, 1], dtype=np.int64)

    return select


def ppo_selector(model: PPO) -> ActionSelector:
    def select(observation: np.ndarray, inventory: float, seed: int) -> np.ndarray:
        del inventory, seed
        return np.asarray(model.predict(observation, deterministic=True)[0], dtype=np.int64)

    return select


def evaluate_actor(
    selector: ActionSelector,
    *,
    orderbook_dir: Path,
    trades_dir: Path,
    windows: list[EvaluationWindow],
    maker_fee: float = 0.0002,
    mandatory_quoting: bool = False,
    queue_fraction: float = 0.5,
    quote_spread_bps: float = 5.0,
    inventory_penalty_multiplier: float = 1.0,
    size_multipliers: tuple[float, float, float] = (0.5, 1.0, 2.0),
    quote_offset_scale: float = 1.0,
    residual_continuous: bool = False,
    residual_variant: str = "legacy",
    base_parameters: BaselineParameters | None = None,
) -> EvaluationSummary:
    episodes = [
        evaluate_window(
            selector,
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            window=window,
            maker_fee=maker_fee,
            mandatory_quoting=mandatory_quoting,
            queue_fraction=queue_fraction,
            quote_spread_bps=quote_spread_bps,
            inventory_penalty_multiplier=inventory_penalty_multiplier,
            size_multipliers=size_multipliers,
            quote_offset_scale=quote_offset_scale,
            residual_continuous=residual_continuous,
            residual_variant=residual_variant,
            base_parameters=base_parameters,
        )
        for window in windows
    ]
    summaries = {}
    for metric_field in fields(EpisodeMetrics):
        values = np.asarray(
            [float(getattr(episode, metric_field.name)) for episode in episodes],
            dtype=float,
        )
        summaries[metric_field.name] = MeanStd(float(np.mean(values)), float(np.std(values)))
    return EvaluationSummary(
        **summaries,
        median_total_pnl=float(np.median([episode.total_pnl for episode in episodes])),
        maximum_observed_inventory=max(
            (episode.max_abs_inventory for episode in episodes),
            default=0.0,
        ),
        window_ids=tuple(window.identifier for window in windows),
    )


def evaluate_window(
    selector: ActionSelector,
    *,
    orderbook_dir: Path,
    trades_dir: Path,
    window: EvaluationWindow,
    maker_fee: float = 0.0002,
    mandatory_quoting: bool = False,
    queue_fraction: float = 0.5,
    quote_spread_bps: float = 5.0,
    inventory_penalty_multiplier: float = 1.0,
    size_multipliers: tuple[float, float, float] = (0.5, 1.0, 2.0),
    quote_offset_scale: float = 1.0,
    residual_continuous: bool = False,
    residual_variant: str = "legacy",
    base_parameters: BaselineParameters | None = None,
) -> EpisodeMetrics:
    day = window.day.isoformat()
    env = RealOrderbookEnv(
        data_dir=orderbook_dir,
        trades_dir=trades_dir,
        start_date=day,
        end_date=day,
        episode_steps=window.steps,
        seed=window.seed,
        maker_fee=maker_fee,
        mandatory_quoting=mandatory_quoting,
        queue_fraction=queue_fraction,
        quote_spread_bps=quote_spread_bps,
        inventory_penalty_multiplier=inventory_penalty_multiplier,
        size_multipliers=size_multipliers,
        quote_offset_scale=quote_offset_scale,
        residual_continuous=residual_continuous,
        residual_variant=residual_variant,
        base_spread_bps=(base_parameters.spread_bps if base_parameters else 5.0),
        base_volatility_filter=(
            base_parameters.volatility_filter if base_parameters else False
        ),
        base_imbalance_filter=(
            base_parameters.imbalance_filter if base_parameters else False
        ),
    )
    observation, _ = env.reset(
        seed=window.seed,
        options={"date": day, "start_index": window.start_index},
    )
    equity_curve = [env.initial_cash]
    step_returns = []
    inventories = []
    total_reward = 0.0
    quoted_steps = 0
    fills = 0
    steps = 0
    markout_1s = []
    markout_10s = []
    truncated = False
    while not truncated:
        previous_equity = env.equity
        action = selector(observation, env.inventory, window.seed)
        observation, reward, _, truncated, info = env.step(action)
        total_reward += reward
        inventories.append(abs(env.inventory))
        equity_curve.append(env.equity)
        step_returns.append((env.equity - previous_equity) / previous_equity)
        quoted_steps += int(info["quote_active"])
        fills += int(info["bid_filled"]) + int(info["ask_filled"])
        if info["bid_filled"] or info["ask_filled"]:
            markout_1s.append(float(info["markout_1s"]))
            markout_10s.append(float(info["markout_10s"]))
        steps += 1
    episode_pnl = env.equity - env.initial_cash
    returns = np.asarray(step_returns, dtype=float)
    return_std = float(np.std(returns))
    sharpe = (
        float(np.mean(returns) / return_std * np.sqrt(len(returns)))
        if return_std > 1e-12
        else 0.0
    )
    curve = np.asarray(equity_curve, dtype=float)
    maximum_drawdown = float(np.max(np.maximum.accumulate(curve) - curve))
    metrics = EpisodeMetrics(
        total_pnl=episode_pnl,
        gross_pnl_before_fees=episode_pnl + env.total_fees,
        total_fees=env.total_fees,
        total_reward=total_reward,
        sharpe_ratio=sharpe,
        maximum_drawdown=maximum_drawdown,
        max_abs_inventory=max(inventories, default=0.0),
        mean_abs_inventory=float(np.mean(inventories)) if inventories else 0.0,
        final_inventory=env.inventory,
        quote_rate=quoted_steps / steps if steps else 0.0,
        fill_rate=fills / steps if steps else 0.0,
        total_turnover=env.total_turnover,
        markout_1s=float(np.mean(markout_1s)) if markout_1s else 0.0,
        markout_10s=float(np.mean(markout_10s)) if markout_10s else 0.0,
        profitable_episode_percentage=100.0 if episode_pnl > 0.0 else 0.0,
    )
    env.close()
    return metrics


def print_metrics_table(results: dict[str, EvaluationSummary]) -> None:
    metric_names = [field.name for field in fields(EpisodeMetrics)]
    headers = ["strategy", *metric_names]
    rows = []
    for name, summary in results.items():
        row = {"strategy": name}
        for metric_name in metric_names:
            value = getattr(summary, metric_name)
            row[metric_name] = f"{value.mean:.4f} +/- {value.std:.4f}"
        rows.append(row)
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows)) for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(row[header].ljust(widths[header]) for header in headers))


def print_profitability(results: dict[str, EvaluationSummary]) -> None:
    print("profitable before fees:")
    for name, summary in results.items():
        profitable = summary.gross_pnl_before_fees.mean > 0.0
        percentage = summary.profitable_episode_percentage.mean
        print(
            f"  {name}: {str(profitable).lower()} "
            f"(profitable_episodes={percentage:.2f}%)"
        )


def print_previous_comparison(
    results: dict[str, EvaluationSummary],
    selected_seed: int,
) -> None:
    current_names = {
        "adaptive_fixed_spread": "adaptive_fixed_spread",
        "adaptive_inventory_skew": "adaptive_inventory_skew",
        "selected_ppo": f"selected_ppo_seed{selected_seed}",
    }
    print("comparison with previous maker_fee=0.0002 experiment:")
    for reference_name, current_name in current_names.items():
        previous = PREVIOUS_FEE_EXPERIMENT[reference_name]
        current = results[current_name]
        print(
            f"  {reference_name}: previous_total_pnl={previous['total_pnl']:.4f}, "
            f"current_total_pnl={current.total_pnl.mean:.4f}, "
            f"previous_gross_pnl={previous['gross_pnl']:.4f}, "
            f"current_gross_pnl={current.gross_pnl_before_fees.mean:.4f}"
        )


def iter_dates(start_date: str, end_date: str):
    current = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if end < current:
        raise ValueError("end_date must be on or after start_date")
    while current <= end:
        yield current
        current += timedelta(days=1)


if __name__ == "__main__":
    main()
