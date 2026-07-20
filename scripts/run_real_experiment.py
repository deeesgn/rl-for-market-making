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
from stable_baselines3.common.utils import polyak_update, safe_mean
from torch import nn
from torch.nn import functional as F

from rl_mm.data.trades import prepare_trade_range
from rl_mm.env import RealOrderbookEnv
from rl_mm.strategies import (
    AvellanedaStoikovStrategy,
    BaseStrategy,
    FixedSpreadStrategy,
    InventorySkewStrategy,
)


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
    active_quote_seconds: float
    active_quote_side_seconds: float
    fill_event_count: float
    filled_side_count: float
    filled_base_quantity: float
    submitted_base_quantity: float
    fill_event_rate: float
    quoted_side_fill_rate: float
    volume_fill_ratio: float
    total_turnover: float
    orders_created: float
    orders_preserved: float
    orders_replaced: float
    orders_cancelled: float
    average_order_age_seconds: float
    average_queue_ahead_at_fill: float
    spread_capture: float
    side_adjusted_markout_1s: float
    side_adjusted_markout_5s: float
    side_adjusted_markout_30s: float
    adverse_selection_contribution: float
    inventory_mark_to_market_contribution: float
    realized_plus_terminal_inventory_pnl: float
    total_equity_pnl: float
    pnl_reconciliation_error: float
    pnl_per_filled_btc: float
    markout_5s_per_filled_btc: float
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
    active_quote_seconds: MeanStd
    active_quote_side_seconds: MeanStd
    fill_event_count: MeanStd
    filled_side_count: MeanStd
    filled_base_quantity: MeanStd
    submitted_base_quantity: MeanStd
    fill_event_rate: MeanStd
    quoted_side_fill_rate: MeanStd
    volume_fill_ratio: MeanStd
    total_turnover: MeanStd
    orders_created: MeanStd
    orders_preserved: MeanStd
    orders_replaced: MeanStd
    orders_cancelled: MeanStd
    average_order_age_seconds: MeanStd
    average_queue_ahead_at_fill: MeanStd
    spread_capture: MeanStd
    side_adjusted_markout_1s: MeanStd
    side_adjusted_markout_5s: MeanStd
    side_adjusted_markout_30s: MeanStd
    adverse_selection_contribution: MeanStd
    inventory_mark_to_market_contribution: MeanStd
    realized_plus_terminal_inventory_pnl: MeanStd
    total_equity_pnl: MeanStd
    pnl_reconciliation_error: MeanStd
    pnl_per_filled_btc: MeanStd
    markout_5s_per_filled_btc: MeanStd
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
class AvellanedaStoikovParameters:
    gamma: float
    k: float


@dataclass(frozen=True)
class FillDiagnosticConfiguration:
    spread_bps: float
    imbalance_strength: float
    queue_fraction: float = 0.25

    @property
    def name(self) -> str:
        return (
            f"spread={self.spread_bps:g}_imbalance={self.imbalance_strength:.2f}_"
            f"queue={self.queue_fraction:.2f}"
        )


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
AS_GAMMA_CANDIDATES = (0.001, 0.005, 0.01)
AS_K_CANDIDATES = (0.1, 0.5, 1.0)
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
RESIDUAL_EARLY_STOP_VARIANT = "selective_narrow"
RESIDUAL_VALIDATION_INTERVAL = 50_000
RESIDUAL_EARLY_STOP_PATIENCE = 3
RESIDUAL_MIN_SCORE_IMPROVEMENT = 0.01
RESIDUAL_ACTION_L2_COEFFICIENT = 0.01
RESIDUAL_BC_COEFFICIENT_START = 0.05
RESIDUAL_BC_COEFFICIENT_END = 0.005
RESIDUAL_EARLY_STOP_STATE = Path("models/search_tmp/residual_sac_early_stop_state.json")


class AnchoredSAC(SAC):
    """SAC with actor-only anchors toward baseline residual actions."""

    def __init__(
        self,
        *args: object,
        action_l2_coefficient: float = RESIDUAL_ACTION_L2_COEFFICIENT,
        bc_coefficient_start: float = RESIDUAL_BC_COEFFICIENT_START,
        bc_coefficient_end: float = RESIDUAL_BC_COEFFICIENT_END,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if action_l2_coefficient < 0.0:
            raise ValueError("action_l2_coefficient must be non-negative")
        if bc_coefficient_start < 0.0 or bc_coefficient_end < 0.0:
            raise ValueError("behaviour-cloning coefficients must be non-negative")
        self.action_l2_coefficient = float(action_l2_coefficient)
        self.bc_coefficient_start = float(bc_coefficient_start)
        self.bc_coefficient_end = float(bc_coefficient_end)
        self.anchor_total_timesteps = 1
        self.anchor_observations: torch.Tensor | None = None
        self.anchor_actions: torch.Tensor | None = None

    def _excluded_save_params(self) -> list[str]:
        return [
            *super()._excluded_save_params(),
            "anchor_observations",
            "anchor_actions",
        ]

    def set_actor_anchors(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        total_timesteps: int,
    ) -> None:
        if len(observations) != len(actions) or len(observations) == 0:
            raise ValueError("Actor anchors require equally sized non-empty batches")
        if total_timesteps < 1:
            raise ValueError("total_timesteps must be positive")
        self.anchor_observations = torch.as_tensor(observations, device=self.device)
        self.anchor_actions = torch.as_tensor(actions, device=self.device)
        self.anchor_total_timesteps = int(total_timesteps)

    def anchor_coefficients(self) -> tuple[float, float]:
        progress = min(max(self.num_timesteps / self.anchor_total_timesteps, 0.0), 1.0)
        cloning = self.bc_coefficient_start + progress * (
            self.bc_coefficient_end - self.bc_coefficient_start
        )
        return self.action_l2_coefficient, cloning

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        ent_coef_losses: list[float] = []
        ent_coefs: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        action_l2_losses: list[float] = []
        cloning_losses: list[float] = []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env,
            )
            discounts = (
                replay_data.discounts if replay_data.discounts is not None else self.gamma
            )
            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(
                    self.log_ent_coef * (log_prob + self.target_entropy).detach()
                ).mean()
                ent_coef_losses.append(float(ent_coef_loss.item()))
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(float(ent_coef.item()))

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = torch.cat(
                    self.critic_target(replay_data.next_observations, next_actions), dim=1
                )
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (
                    1 - replay_data.dones
                ) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values) for current_q in current_q_values
            )
            critic_losses.append(float(critic_loss.item()))
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            action_l2_coefficient, cloning_coefficient = self.anchor_coefficients()
            neutral_action = torch.zeros_like(actions_pi)
            action_l2_loss = torch.mean((actions_pi - neutral_action) ** 2)
            actor_loss = actor_loss + action_l2_coefficient * action_l2_loss
            action_l2_losses.append(float(action_l2_loss.item()))
            cloning_loss = torch.zeros((), device=self.device)
            if self.anchor_observations is not None and self.anchor_actions is not None:
                anchor_indices = torch.randint(
                    len(self.anchor_observations),
                    (min(batch_size, len(self.anchor_observations)),),
                    device=self.device,
                )
                predicted_anchor_actions = self.actor(
                    self.anchor_observations[anchor_indices], deterministic=True
                )
                cloning_loss = F.mse_loss(
                    predicted_anchor_actions,
                    self.anchor_actions[anchor_indices],
                )
                actor_loss = actor_loss + cloning_coefficient * cloning_loss
            cloning_losses.append(float(cloning_loss.item()))
            actor_losses.append(float(actor_loss.item()))
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", safe_mean(ent_coefs))
        self.logger.record("train/actor_loss", safe_mean(actor_losses))
        self.logger.record("train/critic_loss", safe_mean(critic_losses))
        self.logger.record("train/action_l2_loss", safe_mean(action_l2_losses))
        self.logger.record("train/cloning_loss", safe_mean(cloning_losses))
        _, cloning_coefficient = self.anchor_coefficients()
        self.logger.record("train/action_l2_coefficient", self.action_l2_coefficient)
        self.logger.record("train/cloning_coefficient", cloning_coefficient)
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss", safe_mean(ent_coef_losses))


def load_residual_sac_model(
    path: Path,
    *,
    env: RealOrderbookEnv | None = None,
    anchored: bool = False,
) -> SAC:
    """Load residual archives with the repository-local temporal extractor."""
    policy_kwargs = {
        "features_extractor_class": TemporalCNNExtractor,
        "features_extractor_kwargs": {"features_dim": 32},
        "net_arch": [64, 64],
        "use_sde": False,
    }
    algorithm = AnchoredSAC if anchored else SAC
    return algorithm.load(
        path,
        env=env,
        device="cpu",
        custom_objects={"policy_kwargs": policy_kwargs},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete real-data experiment.")
    parser.add_argument(
        "--mode",
        choices=[
            "prepare",
            "train",
            "evaluate",
            "all",
            "full",
            "optimize",
            "residual",
            "residual-early-stop",
            "diagnose-fills",
            "final-as-comparison",
        ],
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
    parser.add_argument("--validation-interval", type=int, default=RESIDUAL_VALIDATION_INTERVAL)
    parser.add_argument("--early-stop-patience", type=int, default=RESIDUAL_EARLY_STOP_PATIENCE)
    parser.add_argument(
        "--min-score-improvement",
        type=float,
        default=RESIDUAL_MIN_SCORE_IMPROVEMENT,
    )
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
    if args.mode in {"residual", "residual-early-stop"} and not np.isclose(
        args.maker_fee, 0.0
    ):
        raise ValueError("The first residual SAC experiment requires --maker-fee 0")
    if args.mode in {"residual", "residual-early-stop", "diagnose-fills"}:
        args.queue_fraction = SEARCH_QUEUE_FRACTION
    if args.mode == "final-as-comparison":
        if not np.isclose(args.maker_fee, 0.0):
            raise ValueError("final AS comparison requires --maker-fee 0")
        args.queue_fraction = SEARCH_QUEUE_FRACTION
    if args.mode == "residual-early-stop":
        if args.validation_interval < 1:
            raise ValueError("--validation-interval must be positive")
        if args.early_stop_patience < 1:
            raise ValueError("--early-stop-patience must be positive")
        if args.min_score_improvement < 0.0:
            raise ValueError("--min-score-improvement must be non-negative")
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

    if args.mode == "residual-early-stop":
        run_residual_early_stop_experiment(
            orderbook_dir=args.orderbook_dir,
            trades_dir=trades_symbol_dir,
            split=split,
            seeds=seeds,
            maker_fee=args.maker_fee,
            full_timesteps=args.full_timesteps,
            validation_interval=args.validation_interval,
            early_stop_patience=args.early_stop_patience,
            min_score_improvement=args.min_score_improvement,
            bc_samples=args.bc_samples,
            force_train=args.force_train,
        )
        return

    if args.mode == "diagnose-fills":
        run_real_fill_diagnostics(
            orderbook_dir=args.orderbook_dir,
            trades_dir=trades_symbol_dir,
            split=split,
            maker_fee=args.maker_fee,
        )
        return

    if args.mode == "final-as-comparison":
        run_final_as_comparison(
            orderbook_dir=args.orderbook_dir,
            trades_dir=trades_symbol_dir,
            split=split,
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


@dataclass(frozen=True)
class ResidualEarlyStopResult:
    best_path: Path | None
    best_timestep: int | None
    stopped_early: bool
    imitation: tuple[float, float] | None


def residual_early_stop_paths(
    seed: int,
    *,
    search_dir: Path = Path("models/search_tmp"),
) -> tuple[Path, Path]:
    prefix = f"residual_sac_{RESIDUAL_EARLY_STOP_VARIANT}_seed{seed}"
    return search_dir / f"{prefix}_resume.zip", search_dir / f"{prefix}_best.zip"


def initial_residual_early_stop_state(seeds: list[int]) -> dict[str, object]:
    return {
        "version": 1,
        "status": "running",
        "variant": RESIDUAL_EARLY_STOP_VARIANT,
        "seeds": {
            str(seed): {
                "status": "pending",
                "timesteps": 0,
                "best_score": None,
                "best_timestep": None,
                "checks_without_improvement": 0,
                "history": [],
            }
            for seed in seeds
        },
    }


def load_residual_early_stop_state(path: Path, seeds: list[int]) -> dict[str, object]:
    if not path.is_file():
        return initial_residual_early_stop_state(seeds)
    with path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    if state.get("version") != 1 or state.get("variant") != RESIDUAL_EARLY_STOP_VARIANT:
        raise ValueError("Unsupported residual early-stop state")
    state.setdefault("seeds", {})
    for seed in seeds:
        state["seeds"].setdefault(
            str(seed),
            {
                "status": "pending",
                "timesteps": 0,
                "best_score": None,
                "best_timestep": None,
                "checks_without_improvement": 0,
                "history": [],
            },
        )
    return state


def save_residual_early_stop_state(path: Path, state: dict[str, object]) -> None:
    save_search_state(path, state)


def checkpoint_improves_score(
    score: float,
    best_score: float | None,
    minimum_improvement: float,
) -> bool:
    if not np.isfinite(score):
        return False
    return best_score is None or score >= best_score + minimum_improvement


def update_early_stop_progress(
    *,
    score: float,
    best_score: float | None,
    checks_without_improvement: int,
    minimum_improvement: float,
    patience: int,
) -> tuple[bool, int, bool]:
    improved = checkpoint_improves_score(score, best_score, minimum_improvement)
    next_checks = 0 if improved else checks_without_improvement + 1
    return improved, next_checks, next_checks >= patience


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


def train_residual_sac_early_stop(
    *,
    orderbook_dir: Path,
    trades_dir: Path,
    train_dates: tuple[date, ...],
    validation_windows: list[EvaluationWindow],
    base_parameters: BaselineParameters,
    baseline_summary: EvaluationSummary,
    seed: int,
    target_timesteps: int,
    validation_interval: int,
    patience: int,
    minimum_score_improvement: float,
    bc_samples: int,
    state: dict[str, object],
    state_path: Path,
    search_dir: Path = Path("models/search_tmp"),
) -> ResidualEarlyStopResult:
    """Train one anchored SAC seed and retain only its best validation model."""
    if not train_dates:
        raise ValueError("Residual SAC training requires train dates")
    if validation_interval < 1 or patience < 1 or minimum_score_improvement < 0.0:
        raise ValueError("Invalid early-stopping configuration")

    seed_state = state["seeds"][str(seed)]
    resume_path, best_path = residual_early_stop_paths(seed, search_dir=search_dir)
    completed_statuses = {"complete", "early_stopped"}
    if seed_state.get("status") in completed_statuses:
        recorded_best = Path(str(seed_state.get("best_checkpoint", best_path)))
        if recorded_best.is_file():
            return ResidualEarlyStopResult(
                best_path=recorded_best,
                best_timestep=int(seed_state["best_timestep"]),
                stopped_early=seed_state.get("status") == "early_stopped",
                imitation=None,
            )

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
        residual_variant=RESIDUAL_EARLY_STOP_VARIANT,
        base_spread_bps=base_parameters.spread_bps,
        base_volatility_filter=base_parameters.volatility_filter,
        base_imbalance_filter=base_parameters.imbalance_filter,
    )
    imitation = None
    try:
        examples, expert_actions, used_dates = collect_behavior_cloning_batch(
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            train_dates=train_dates,
            base_parameters=base_parameters,
            sample_count=bc_samples,
            seed=seed,
            residual_variant=RESIDUAL_EARLY_STOP_VARIANT,
        )
        if not set(used_dates).issubset(train_dates):
            raise RuntimeError("Behaviour cloning accessed a non-train date")

        if resume_path.is_file():
            model = load_residual_sac_model(
                resume_path,
                env=env,
                anchored=True,
            )
            completed_timesteps = max(
                int(seed_state.get("timesteps", 0)),
                int(model.num_timesteps),
            )
        else:
            model = AnchoredSAC(
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
            imitation = pretrain_residual_actor(model, examples, expert_actions)
            print(
                f"baseline_imitation_error variant={RESIDUAL_EARLY_STOP_VARIANT} "
                f"seed={seed}: before={imitation[0]:.6f} after={imitation[1]:.6f}"
            )
            completed_timesteps = 0

        model.set_actor_anchors(
            examples,
            expert_actions,
            total_timesteps=target_timesteps,
        )
        print(
            f"actor_anchoring seed={seed} "
            f"action_l2_coefficient={model.action_l2_coefficient:.6f} "
            f"cloning_coefficient_start={model.bc_coefficient_start:.6f} "
            f"cloning_coefficient_end={model.bc_coefficient_end:.6f}"
        )
        best_score = seed_state.get("best_score")
        best_score = float(best_score) if best_score is not None else None
        checks_without_improvement = int(seed_state.get("checks_without_improvement", 0))
        history = seed_state.setdefault("history", [])

        while completed_timesteps < target_timesteps:
            chunk = min(validation_interval, target_timesteps - completed_timesteps)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            completed_timesteps += chunk
            atomic_save_model(model, resume_path)
            summary = evaluate_actor(
                residual_selector(model),
                orderbook_dir=orderbook_dir,
                trades_dir=trades_dir,
                windows=validation_windows,
                maker_fee=0.0,
                queue_fraction=base_parameters.queue_fraction,
                quote_spread_bps=base_parameters.spread_bps,
                residual_continuous=True,
                residual_variant=RESIDUAL_EARLY_STOP_VARIANT,
                base_parameters=base_parameters,
            )
            components = validation_score_components(summary)
            validation = residual_validation_result(
                summary,
                baseline_summary=baseline_summary,
            )
            improved, checks_without_improvement, stopped_early = update_early_stop_progress(
                score=(
                    components["score"]
                    if bool(validation["valid_metrics"])
                    else float("nan")
                ),
                best_score=best_score,
                checks_without_improvement=checks_without_improvement,
                minimum_improvement=minimum_score_improvement,
                patience=patience,
            )
            if improved:
                best_score = components["score"]
                atomic_save_model(model, best_path)
                seed_state["best_score"] = best_score
                seed_state["best_timestep"] = completed_timesteps
                seed_state["best_checkpoint"] = str(best_path)
            else:
                stopped_early = checks_without_improvement >= patience
            action_l2_coefficient, cloning_coefficient = model.anchor_coefficients()
            history.append(
                {
                    "timestep": completed_timesteps,
                    "mean_validation_pnl": components["mean_validation_pnl"],
                    "std_validation_pnl": components["std_validation_pnl"],
                    "mean_max_drawdown": components["mean_max_drawdown"],
                    "quote_rate": summary.quote_rate.mean,
                    "fill_rate": summary.fill_rate.mean,
                    "score": components["score"],
                    "new_best": improved,
                    "action_l2_coefficient": action_l2_coefficient,
                    "cloning_coefficient": cloning_coefficient,
                }
            )
            seed_state.update(
                {
                    "timesteps": completed_timesteps,
                    "checks_without_improvement": checks_without_improvement,
                    "status": "early_stopped" if stopped_early else "running",
                    "resume_checkpoint": str(resume_path),
                }
            )
            save_residual_early_stop_state(state_path, state)
            print(
                f"validation_checkpoint seed={seed} timestep={completed_timesteps} "
                f"mean_pnl={components['mean_validation_pnl']:.6f} "
                f"pnl_std={components['std_validation_pnl']:.6f} "
                f"mean_max_drawdown={components['mean_max_drawdown']:.6f} "
                f"quote_rate={summary.quote_rate.mean:.6f} "
                f"fill_rate={summary.fill_rate.mean:.6f} score={components['score']:.6f} "
                f"new_best={improved} "
                f"action_l2_coefficient={action_l2_coefficient:.6f} "
                f"cloning_coefficient={cloning_coefficient:.6f}"
            )
            if stopped_early:
                break

        if seed_state.get("status") != "early_stopped":
            seed_state["status"] = "complete"
            save_residual_early_stop_state(state_path, state)
        recorded_best = Path(str(seed_state.get("best_checkpoint", best_path)))
        return ResidualEarlyStopResult(
            best_path=recorded_best if recorded_best.is_file() else None,
            best_timestep=(
                int(seed_state["best_timestep"])
                if seed_state.get("best_timestep") is not None
                else None
            ),
            stopped_early=seed_state.get("status") == "early_stopped",
            imitation=imitation,
        )
    finally:
        env.close()


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
        model = load_residual_sac_model(warm_start_path, env=env)
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
    residual_model = load_residual_sac_model(RESIDUAL_FINAL_MODEL)
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


def build_residual_validation_windows(
    *,
    orderbook_dir: Path,
    split: DatasetSplit,
) -> list[EvaluationWindow]:
    return build_evaluation_windows(
        orderbook_dir=orderbook_dir,
        symbol="BTCUSDT",
        dates=split.validation,
        seeds=[42],
        minimum_episodes=len(split.validation),
    )


def run_residual_early_stop_experiment(
    *,
    orderbook_dir: Path,
    trades_dir: Path,
    split: DatasetSplit,
    seeds: list[int],
    maker_fee: float,
    full_timesteps: int,
    validation_interval: int,
    early_stop_patience: int,
    min_score_improvement: float,
    bc_samples: int,
    force_train: bool,
    state_path: Path = RESIDUAL_EARLY_STOP_STATE,
) -> None:
    """Run the fixed selective-narrow residual SAC experiment with early stopping."""
    del force_train
    if not np.isclose(maker_fee, 0.0):
        raise ValueError("Residual SAC early stopping is currently defined for zero fees")
    state = load_residual_early_stop_state(state_path, seeds)
    if state.get("status") == "complete":
        print(f"residual early-stop experiment already complete: {state_path}")
        return

    base_parameters = BaselineParameters(15.0, SEARCH_QUEUE_FRACTION, False, True)
    validation_windows = build_residual_validation_windows(
        orderbook_dir=orderbook_dir,
        split=split,
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

    validation_summaries: dict[int, EvaluationSummary] = {}
    best_paths: dict[int, Path] = {}
    best_timesteps: dict[int, int] = {}
    for seed in seeds:
        result = train_residual_sac_early_stop(
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            train_dates=split.train,
            validation_windows=validation_windows,
            base_parameters=base_parameters,
            baseline_summary=baseline_summary,
            seed=seed,
            target_timesteps=full_timesteps,
            validation_interval=validation_interval,
            patience=early_stop_patience,
            minimum_score_improvement=min_score_improvement,
            bc_samples=bc_samples,
            state=state,
            state_path=state_path,
        )
        if result.best_path is None or result.best_timestep is None:
            print(f"seed={seed} has no valid validation checkpoint")
            continue
        summary = evaluate_residual_model(
            result.best_path,
            variant=RESIDUAL_EARLY_STOP_VARIANT,
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            windows=validation_windows,
            base_parameters=base_parameters,
        )
        validation_summaries[seed] = summary
        best_paths[seed] = result.best_path
        best_timesteps[seed] = result.best_timestep
        seed_state = state["seeds"][str(seed)]
        seed_state["best_validation_score"] = validation_score_components(summary)["score"]
        seed_state["best_validation_pnl"] = summary.total_pnl.mean
        save_residual_early_stop_state(state_path, state)
        print_validation_score(
            f"{RESIDUAL_EARLY_STOP_VARIANT}_seed{seed}_best_t{result.best_timestep}",
            summary,
            baseline_summary,
        )
        print(
            f"early_stop_result seed={seed} best_checkpoint={result.best_path} "
            f"best_timestep={result.best_timestep} stopped_early={result.stopped_early}"
        )

    selected_seed = select_residual_seed(
        validation_summaries,
        baseline_summary=baseline_summary,
    )
    if selected_seed is None:
        state["status"] = "complete"
        state["selected_seed"] = None
        save_residual_early_stop_state(state_path, state)
        print("selected_residual_seed: none")
        print("test_evaluation_triggered: false")
        return

    selected_path = best_paths[selected_seed]
    atomic_copy_model(selected_path, RESIDUAL_FINAL_MODEL)
    state["status"] = "complete"
    state["selected_seed"] = selected_seed
    state["selected_best_timestep"] = best_timesteps[selected_seed]
    state["selected_checkpoint"] = str(selected_path)
    state["final_model_path"] = str(RESIDUAL_FINAL_MODEL)
    save_residual_early_stop_state(state_path, state)
    print(f"selected_residual_seed: {selected_seed}")
    print(f"selected_best_timestep: {best_timesteps[selected_seed]}")
    print(f"best_residual_model: {RESIDUAL_FINAL_MODEL}")

    test_windows = build_residual_test_windows(
        orderbook_dir=orderbook_dir,
        split=split,
        promotion_complete=True,
    )
    previous_model, previous_candidate = previous_ppo_settings()
    fixed_actor = baseline_actor("fixed", base_parameters, False)
    residual_model = load_residual_sac_model(RESIDUAL_FINAL_MODEL)
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
            residual_variant=RESIDUAL_EARLY_STOP_VARIANT,
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
    model = load_residual_sac_model(path)
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


def avellaneda_stoikov_candidates() -> tuple[AvellanedaStoikovParameters, ...]:
    return tuple(
        AvellanedaStoikovParameters(gamma=gamma, k=k)
        for gamma in AS_GAMMA_CANDIDATES
        for k in AS_K_CANDIDATES
    )


def select_avellaneda_stoikov_parameters(
    validation_results: dict[AvellanedaStoikovParameters, EvaluationSummary],
) -> AvellanedaStoikovParameters:
    """Select AS parameters from validation metrics only."""
    if not validation_results:
        raise ValueError("AS validation results are required for parameter selection")
    return max(
        validation_results,
        key=lambda parameters: (
            validation_score_components(validation_results[parameters])["score"],
            -parameters.gamma,
            -parameters.k,
        ),
    )


def evaluate_avellaneda_stoikov(
    parameters: AvellanedaStoikovParameters,
    *,
    orderbook_dir: Path,
    trades_dir: Path,
    windows: list[EvaluationWindow],
    maker_fee: float = 0.0,
    queue_fraction: float = SEARCH_QUEUE_FRACTION,
) -> EvaluationSummary:
    episodes = [
        evaluate_window(
            None,
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            window=window,
            maker_fee=maker_fee,
            queue_fraction=queue_fraction,
            avellaneda_stoikov=AvellanedaStoikovStrategy(
                gamma=parameters.gamma,
                k=parameters.k,
            ),
        )
        for window in windows
    ]
    return aggregate_evaluation(episodes, windows)


def tune_avellaneda_stoikov(
    *,
    orderbook_dir: Path,
    trades_dir: Path,
    windows: list[EvaluationWindow],
) -> tuple[
    AvellanedaStoikovParameters,
    dict[AvellanedaStoikovParameters, EvaluationSummary],
]:
    results: dict[AvellanedaStoikovParameters, EvaluationSummary] = {}
    print("Avellaneda-Stoikov validation (36 fixed one-hour windows):")
    for parameters in avellaneda_stoikov_candidates():
        summary = evaluate_avellaneda_stoikov(
            parameters,
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            windows=windows,
        )
        results[parameters] = summary
        components = validation_score_components(summary)
        print(
            f"  gamma={parameters.gamma:g} k={parameters.k:g} "
            f"mean_pnl={components['mean_validation_pnl']:.6f} "
            f"pnl_std={components['std_validation_pnl']:.6f} "
            f"mean_max_drawdown={components['mean_max_drawdown']:.6f} "
            f"score={components['score']:.6f}"
        )
    return select_avellaneda_stoikov_parameters(results), results


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
    imbalance_strength: float = 0.0,
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
            imbalance_strength=imbalance_strength,
        )
        for window in windows
    ]
    return aggregate_evaluation(episodes, windows)


def aggregate_evaluation(
    episodes: list[EpisodeMetrics],
    windows: list[EvaluationWindow],
) -> EvaluationSummary:
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
    selector: ActionSelector | None,
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
    imbalance_strength: float = 0.0,
    avellaneda_stoikov: AvellanedaStoikovStrategy | None = None,
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
        imbalance_strength=imbalance_strength,
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
        if avellaneda_stoikov is None:
            if selector is None:
                raise ValueError("selector is required when AS strategy is not provided")
            action = selector(observation, env.inventory, window.seed)
            observation, reward, _, truncated, info = env.step(action)
        else:
            market = env.current_market_state()
            quotes = avellaneda_stoikov.quote(**market)
            observation, reward, _, truncated, info = env.step_quotes(
                bid_price=quotes.bid_price,
                ask_price=quotes.ask_price,
                bid_size=quotes.bid_size,
                ask_size=quotes.ask_size,
            )
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
    execution = env.execution_diagnostics()
    if not np.isclose(execution["pnl_reconciliation_error"], 0.0, atol=1e-8):
        raise RuntimeError(
            "Execution PnL decomposition does not reconcile: "
            f"{execution['pnl_reconciliation_error']:.12f}"
        )
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
        # Backward-compatible metric: filled bid/ask sides divided by episode seconds.
        fill_rate=fills / steps if steps else 0.0,
        active_quote_seconds=float(env.active_quote_seconds),
        active_quote_side_seconds=float(env.active_quote_side_seconds),
        fill_event_count=float(env.fill_event_count),
        filled_side_count=float(env.filled_side_count),
        filled_base_quantity=env.filled_base_quantity,
        submitted_base_quantity=env.submitted_base_quantity,
        fill_event_rate=(
            env.fill_event_count / env.active_quote_seconds
            if env.active_quote_seconds
            else 0.0
        ),
        quoted_side_fill_rate=(
            env.filled_side_count / env.active_quote_side_seconds
            if env.active_quote_side_seconds
            else 0.0
        ),
        volume_fill_ratio=(
            env.filled_base_quantity / env.submitted_base_quantity
            if env.submitted_base_quantity > 0.0
            else 0.0
        ),
        total_turnover=env.total_turnover,
        orders_created=float(env.orders_created),
        orders_preserved=float(env.orders_preserved),
        orders_replaced=float(env.orders_replaced),
        orders_cancelled=float(env.orders_cancelled),
        average_order_age_seconds=(
            env._order_age_seconds_total / env.active_quote_side_seconds
            if env.active_quote_side_seconds
            else 0.0
        ),
        average_queue_ahead_at_fill=(
            env._queue_ahead_at_fill_total / env.filled_side_count
            if env.filled_side_count
            else 0.0
        ),
        spread_capture=execution["spread_capture"],
        side_adjusted_markout_1s=execution["side_adjusted_markout_1s"],
        side_adjusted_markout_5s=execution["side_adjusted_markout_5s"],
        side_adjusted_markout_30s=execution["side_adjusted_markout_30s"],
        adverse_selection_contribution=execution["adverse_selection_contribution"],
        inventory_mark_to_market_contribution=execution[
            "inventory_mark_to_market_contribution"
        ],
        realized_plus_terminal_inventory_pnl=execution[
            "realized_plus_terminal_inventory_pnl"
        ],
        total_equity_pnl=execution["total_equity_pnl"],
        pnl_reconciliation_error=execution["pnl_reconciliation_error"],
        pnl_per_filled_btc=(
            episode_pnl / env.filled_base_quantity
            if env.filled_base_quantity > 0.0
            else 0.0
        ),
        markout_5s_per_filled_btc=(
            execution["side_adjusted_markout_5s"] / env.filled_base_quantity
            if env.filled_base_quantity > 0.0
            else 0.0
        ),
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


def assert_identical_evaluation_windows(
    results: dict[str, EvaluationSummary],
) -> tuple[str, ...]:
    """Reject comparisons that did not use one shared deterministic window set."""
    if not results:
        raise ValueError("At least one evaluation result is required")
    window_sets = {summary.window_ids for summary in results.values()}
    if len(window_sets) != 1:
        raise ValueError("All strategies must use identical evaluation windows")
    return next(iter(window_sets))


def print_final_as_table(
    results: dict[str, EvaluationSummary],
    *,
    hours: int,
) -> None:
    headers = (
        "strategy",
        "mean_pnl",
        "total_pnl",
        "pnl_std",
        "profitable_hours",
        "max_drawdown",
        "quote_rate",
        "fill_rate",
        "turnover",
        "max_inventory",
        "markout_1s",
        "markout_5s",
        "markout_30s",
    )
    rows = []
    for name, summary in results.items():
        profitable_hours = round(
            summary.profitable_episode_percentage.mean * hours / 100.0
        )
        rows.append(
            {
                "strategy": name,
                "mean_pnl": f"{summary.total_pnl.mean:.4f}",
                "total_pnl": f"{summary.total_pnl.mean * hours:.4f}",
                "pnl_std": f"{summary.total_pnl.std:.4f}",
                "profitable_hours": f"{profitable_hours}/{hours}",
                "max_drawdown": f"{summary.maximum_drawdown.mean:.4f}",
                "quote_rate": f"{summary.quote_rate.mean:.6f}",
                "fill_rate": f"{summary.fill_rate.mean:.6f}",
                "turnover": f"{summary.total_turnover.mean:.2f}",
                "max_inventory": f"{summary.maximum_observed_inventory:.4f}",
                "markout_1s": f"{summary.side_adjusted_markout_1s.mean:.4f}",
                "markout_5s": f"{summary.side_adjusted_markout_5s.mean:.4f}",
                "markout_30s": f"{summary.side_adjusted_markout_30s.mean:.4f}",
            }
        )
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(row[header].ljust(widths[header]) for header in headers))


def run_final_as_comparison(
    *,
    orderbook_dir: Path,
    trades_dir: Path,
    split: DatasetSplit,
) -> tuple[AvellanedaStoikovParameters, dict[str, EvaluationSummary]]:
    """Tune AS on validation and evaluate all frozen strategies once on test."""
    if len(split.validation) != 36 or len(split.test) != 55:
        raise ValueError("Final AS comparison requires the fixed 36/55 validation/test split")
    validation_windows = build_evaluation_windows(
        orderbook_dir=orderbook_dir,
        symbol="BTCUSDT",
        dates=split.validation,
        seeds=[42],
        minimum_episodes=36,
    )
    selected_as, _ = tune_avellaneda_stoikov(
        orderbook_dir=orderbook_dir,
        trades_dir=trades_dir,
        windows=validation_windows,
    )
    print(f"selected_AS_parameters: gamma={selected_as.gamma:g} k={selected_as.k:g}")

    test_windows = build_evaluation_windows(
        orderbook_dir=orderbook_dir,
        symbol="BTCUSDT",
        dates=split.test,
        seeds=[42],
        minimum_episodes=55,
    )
    base_parameters = BaselineParameters(15.0, SEARCH_QUEUE_FRACTION, False, True)
    fixed_actor = baseline_actor("fixed", base_parameters, False)
    inventory_actor = baseline_actor("inventory", base_parameters, False)
    previous_ppo, previous_candidate = previous_ppo_settings()
    if not RESIDUAL_FINAL_MODEL.is_file():
        raise FileNotFoundError(f"Missing residual SAC model: {RESIDUAL_FINAL_MODEL}")
    residual_model = load_residual_sac_model(RESIDUAL_FINAL_MODEL)
    results = {
        "adaptive_fixed_spread": evaluate_actor(
            fixed_actor.selector,
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            windows=test_windows,
            maker_fee=0.0,
            queue_fraction=SEARCH_QUEUE_FRACTION,
            quote_spread_bps=base_parameters.spread_bps,
        ),
        "adaptive_inventory_skew": evaluate_actor(
            inventory_actor.selector,
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            windows=test_windows,
            maker_fee=0.0,
            queue_fraction=SEARCH_QUEUE_FRACTION,
            quote_spread_bps=base_parameters.spread_bps,
        ),
        "avellaneda_stoikov": evaluate_avellaneda_stoikov(
            selected_as,
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            windows=test_windows,
        ),
        "previous_ppo": evaluate_actor(
            ppo_selector(previous_ppo),
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
            queue_fraction=SEARCH_QUEUE_FRACTION,
            quote_spread_bps=base_parameters.spread_bps,
            residual_continuous=True,
            residual_variant=RESIDUAL_EARLY_STOP_VARIANT,
            base_parameters=base_parameters,
        ),
    }
    assert_identical_evaluation_windows(results)
    print("baseline_parameters: " + format_baseline_parameters(base_parameters))
    print("maker_fee: 0.000000")
    print(f"test_windows: {len(test_windows)} one-hour windows")
    print("markouts are side-adjusted USDT contributions per one-hour window")
    print_final_as_table(results, hours=len(test_windows))
    print(
        "This is a zero-fee research experiment and is not evidence of live profitability."
    )
    return selected_as, results


def assert_neutral_residual_equivalence(
    *,
    orderbook_dir: Path,
    trades_dir: Path,
    windows: list[EvaluationWindow],
    parameters: BaselineParameters,
    maker_fee: float,
    imbalance_strength: float = 0.0,
) -> None:
    """Fail on the first live-replay difference between baseline and neutral residual."""
    actor = baseline_actor("inventory", parameters, False)
    neutral_action = RealOrderbookEnv.neutral_residual_action("selective_narrow")
    compared_fields = (
        "bid_quote",
        "ask_quote",
        "bid_order_size_btc",
        "ask_order_size_btc",
        "bid_filled",
        "ask_filled",
        "bid_fill_size",
        "ask_fill_size",
        "inventory",
        "total_turnover",
        "total_fees",
        "equity",
        "reward",
    )
    for window in windows:
        day = window.day.isoformat()
        common = {
            "data_dir": orderbook_dir,
            "trades_dir": trades_dir,
            "start_date": day,
            "end_date": day,
            "episode_steps": window.steps,
            "seed": window.seed,
            "maker_fee": maker_fee,
            "queue_fraction": parameters.queue_fraction,
            "quote_spread_bps": parameters.spread_bps,
            "base_spread_bps": parameters.spread_bps,
            "base_volatility_filter": parameters.volatility_filter,
            "base_imbalance_filter": parameters.imbalance_filter,
            "imbalance_strength": imbalance_strength,
        }
        baseline_env = RealOrderbookEnv(**common)
        residual_env = RealOrderbookEnv(
            **common,
            residual_continuous=True,
            residual_variant="selective_narrow",
        )
        try:
            baseline_observation, _ = baseline_env.reset(
                seed=window.seed,
                options={"date": day, "start_index": window.start_index},
            )
            residual_env.reset(
                seed=window.seed,
                options={"date": day, "start_index": window.start_index},
            )
            truncated = False
            while not truncated:
                action = actor.selector(
                    baseline_observation,
                    baseline_env.inventory,
                    window.seed,
                )
                baseline_observation, _, _, truncated, baseline_info = baseline_env.step(
                    action
                )
                _, _, _, residual_truncated, residual_info = residual_env.step(neutral_action)
                if residual_truncated != truncated:
                    raise AssertionError(
                        f"neutral residual termination mismatch on {window.identifier}"
                    )
                for field_name in compared_fields:
                    baseline_value = baseline_info[field_name]
                    residual_value = residual_info[field_name]
                    if isinstance(baseline_value, (float, np.floating)) or isinstance(
                        residual_value, (float, np.floating)
                    ):
                        matches = (
                            baseline_value is None
                            and residual_value is None
                            or baseline_value is not None
                            and residual_value is not None
                            and np.isclose(baseline_value, residual_value, atol=1e-12)
                        )
                    else:
                        matches = baseline_value == residual_value
                    if not matches:
                        raise AssertionError(
                            "neutral residual mismatch "
                            f"window={window.identifier} timestep={baseline_env.step_count} "
                            f"field={field_name} baseline={baseline_value!r} "
                            f"residual={residual_value!r}"
                        )
        finally:
            baseline_env.close()
            residual_env.close()


def run_real_fill_diagnostics(
    *,
    orderbook_dir: Path,
    trades_dir: Path,
    split: DatasetSplit,
    maker_fee: float,
) -> None:
    """Print a validation-only audit of fills, persistence, spread, and imbalance."""
    windows = build_evaluation_windows(
        orderbook_dir=orderbook_dir,
        symbol="BTCUSDT",
        dates=split.validation,
        seeds=[42],
        minimum_episodes=len(split.validation),
    )
    neutral_parameters = BaselineParameters(15.0, 0.25, False, False)
    assert_neutral_residual_equivalence(
        orderbook_dir=orderbook_dir,
        trades_dir=trades_dir,
        windows=windows,
        parameters=neutral_parameters,
        maker_fee=maker_fee,
    )
    print("neutral_residual_equivalence: passed")
    print(
        "legacy_fill_rate: numerator=filled_side_count "
        "(bid_filled + ask_filled), denominator=episode_steps"
    )

    configurations = [
        FillDiagnosticConfiguration(spread, strength)
        for spread in (15.0, 10.0, 7.5, 5.0)
        for strength in (0.0, 0.25, 0.5)
    ]
    primary_results: dict[FillDiagnosticConfiguration, EvaluationSummary] = {}
    for configuration in configurations:
        parameters = BaselineParameters(
            configuration.spread_bps,
            configuration.queue_fraction,
            False,
            False,
        )
        actor = baseline_actor("inventory", parameters, False)
        primary_results[configuration] = evaluate_actor(
            actor.selector,
            orderbook_dir=orderbook_dir,
            trades_dir=trades_dir,
            windows=windows,
            maker_fee=maker_fee,
            queue_fraction=configuration.queue_fraction,
            quote_spread_bps=configuration.spread_bps,
            base_parameters=parameters,
            imbalance_strength=configuration.imbalance_strength,
        )
    print_fill_diagnostic_results("primary validation diagnostics", primary_results, len(windows))
    print_fill_rate_impact(primary_results)

    ranked = sorted(
        configurations,
        key=lambda configuration: validation_score_components(
            primary_results[configuration]
        )["score"],
        reverse=True,
    )[:3]
    sensitivity_results: dict[FillDiagnosticConfiguration, EvaluationSummary] = {}
    for configuration in ranked:
        for queue_fraction in (0.10, 0.25, 0.50):
            sensitivity = FillDiagnosticConfiguration(
                configuration.spread_bps,
                configuration.imbalance_strength,
                queue_fraction,
            )
            parameters = BaselineParameters(
                sensitivity.spread_bps,
                sensitivity.queue_fraction,
                False,
                False,
            )
            actor = baseline_actor("inventory", parameters, False)
            sensitivity_results[sensitivity] = evaluate_actor(
                actor.selector,
                orderbook_dir=orderbook_dir,
                trades_dir=trades_dir,
                windows=windows,
                maker_fee=maker_fee,
                queue_fraction=sensitivity.queue_fraction,
                quote_spread_bps=sensitivity.spread_bps,
                base_parameters=parameters,
                imbalance_strength=sensitivity.imbalance_strength,
            )
    print("queue sensitivity is robustness-only; configurations remain ranked at queue=0.25")
    print_fill_diagnostic_results("queue sensitivity", sensitivity_results, len(windows))


def print_fill_diagnostic_results(
    title: str,
    results: dict[FillDiagnosticConfiguration, EvaluationSummary],
    episode_count: int,
) -> None:
    print(title + ":")
    print(
        "configuration  score  mean_pnl  pnl_std  median_pnl  total_pnl  "
        "drawdown  quote_rate  legacy_fill_rate  event_fill_rate  side_fill_rate  "
        "volume_fill_ratio"
    )
    for configuration, summary in results.items():
        score = validation_score_components(summary)["score"]
        print(
            f"{configuration.name}  {score:.6f}  {summary.total_pnl.mean:.6f}  "
            f"{summary.total_pnl.std:.6f}  {summary.median_total_pnl:.6f}  "
            f"{summary.total_pnl.mean * episode_count:.6f}  "
            f"{summary.maximum_drawdown.mean:.6f}  {summary.quote_rate.mean:.6f}  "
            f"{summary.fill_rate.mean:.6f}  {summary.fill_event_rate.mean:.6f}  "
            f"{summary.quoted_side_fill_rate.mean:.6f}  {summary.volume_fill_ratio.mean:.6f}"
        )
        print(
            "  fills: "
            f"active_quote_seconds={summary.active_quote_seconds.mean:.2f}, "
            f"active_quote_side_seconds={summary.active_quote_side_seconds.mean:.2f}, "
            f"fill_event_count={summary.fill_event_count.mean:.2f}, "
            f"filled_side_count={summary.filled_side_count.mean:.2f}, "
            f"filled_base_quantity={summary.filled_base_quantity.mean:.8f}, "
            f"submitted_base_quantity={summary.submitted_base_quantity.mean:.8f}; "
            "orders: "
            f"created={summary.orders_created.mean:.2f}, "
            f"preserved={summary.orders_preserved.mean:.2f}, "
            f"replaced={summary.orders_replaced.mean:.2f}, "
            f"cancelled={summary.orders_cancelled.mean:.2f}, "
            f"avg_age={summary.average_order_age_seconds.mean:.4f}, "
            f"avg_queue_at_fill={summary.average_queue_ahead_at_fill.mean:.8f}"
        )
        print(
            "  execution: "
            f"spread_capture={summary.spread_capture.mean:.6f}, "
            f"markout_1s={summary.side_adjusted_markout_1s.mean:.6f}, "
            f"markout_5s={summary.side_adjusted_markout_5s.mean:.6f}, "
            f"markout_30s={summary.side_adjusted_markout_30s.mean:.6f}, "
            f"adverse_selection={summary.adverse_selection_contribution.mean:.6f}, "
            f"inventory_contribution={summary.inventory_mark_to_market_contribution.mean:.6f}, "
            f"fees={summary.total_fees.mean:.6f}, "
            "realized_plus_terminal_inventory="
            f"{summary.realized_plus_terminal_inventory_pnl.mean:.6f}, "
            f"equity_pnl={summary.total_equity_pnl.mean:.6f}, "
            f"reconciliation_error={summary.pnl_reconciliation_error.mean:.12f}"
        )
        print(
            "  impact: "
            f"profitable_episodes={summary.profitable_episode_percentage.mean:.2f}%, "
            f"turnover={summary.total_turnover.mean:.6f}, "
            f"max_abs_inventory={summary.max_abs_inventory.mean:.8f}, "
            f"mean_abs_inventory={summary.mean_abs_inventory.mean:.8f}, "
            f"pnl_per_filled_btc={summary.pnl_per_filled_btc.mean:.6f}, "
            f"markout_5s_per_filled_btc={summary.markout_5s_per_filled_btc.mean:.6f}"
        )


def print_fill_rate_impact(
    results: dict[FillDiagnosticConfiguration, EvaluationSummary],
) -> None:
    if len(results) < 2:
        return
    fill_rates = np.asarray(
        [summary.fill_event_rate.mean for summary in results.values()], dtype=float
    )
    labels = {
        "pnl_per_filled_btc": np.asarray(
            [summary.pnl_per_filled_btc.mean for summary in results.values()], dtype=float
        ),
        "markout_5s_per_filled_btc": np.asarray(
            [summary.markout_5s_per_filled_btc.mean for summary in results.values()], dtype=float
        ),
        "drawdown": np.asarray(
            [summary.maximum_drawdown.mean for summary in results.values()], dtype=float
        ),
    }
    print("fill-rate relationship across primary validation configurations:")
    for label, values in labels.items():
        if np.std(fill_rates) <= 1e-12 or np.std(values) <= 1e-12:
            direction = "no measurable variation"
        else:
            correlation = float(np.corrcoef(fill_rates, values)[0, 1])
            if label == "drawdown":
                direction = "worsened" if correlation > 0.0 else "improved"
            else:
                direction = "improved" if correlation > 0.0 else "worsened"
            direction += f" (correlation={correlation:.4f})"
        print(f"  higher fill_event_rate vs {label}: {direction}")


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
