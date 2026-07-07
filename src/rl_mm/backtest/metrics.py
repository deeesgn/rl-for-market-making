"""Simple metrics for mock strategy comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class EpisodeMetrics:
    """Summary metrics for one mock environment episode."""

    total_pnl: float
    total_reward: float
    max_abs_inventory: float
    mean_abs_inventory: float
    inventory_std: float
    final_inventory: float
    number_of_steps: int
    quoted_steps: int
    quote_rate: float
    action_0_no_quote: int
    action_1_narrow: int
    action_2_medium: int
    action_3_wide: int
    action_4_skew_sell: int
    action_5_skew_buy: int
    bid_fills: int
    ask_fills: int
    total_fills: int
    fill_rate: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class MetricSummary:
    """Mean and standard deviation for one metric."""

    mean: float
    std: float


@dataclass(frozen=True)
class AggregateMetrics:
    """Aggregated metrics across multiple episodes."""

    total_pnl: MetricSummary
    total_reward: MetricSummary
    max_abs_inventory: MetricSummary
    mean_abs_inventory: MetricSummary
    inventory_std: MetricSummary
    final_inventory: MetricSummary
    number_of_steps: MetricSummary
    quoted_steps: MetricSummary
    quote_rate: MetricSummary
    action_0_no_quote: MetricSummary
    action_1_narrow: MetricSummary
    action_2_medium: MetricSummary
    action_3_wide: MetricSummary
    action_4_skew_sell: MetricSummary
    action_5_skew_buy: MetricSummary
    bid_fills: MetricSummary
    ask_fills: MetricSummary
    total_fills: MetricSummary
    fill_rate: MetricSummary

    def as_dict(self) -> dict[str, MetricSummary]:
        return asdict(self)


def compute_episode_metrics(
    *,
    rewards: Sequence[float],
    pnls: Sequence[float],
    inventories: Sequence[float],
    quoted: Sequence[bool] | None = None,
    actions: Sequence[int] | None = None,
    bid_fills: Sequence[bool] | None = None,
    ask_fills: Sequence[bool] | None = None,
) -> EpisodeMetrics:
    """Compute comparison metrics from one completed episode."""

    total_pnl = float(pnls[-1]) if pnls else 0.0
    total_reward = float(sum(rewards))
    inventory_values = np.array([float(value) for value in inventories], dtype=float)
    abs_inventory = np.abs(inventory_values)
    max_abs_inventory = float(np.max(abs_inventory)) if abs_inventory.size else 0.0
    mean_abs_inventory = float(np.mean(abs_inventory)) if abs_inventory.size else 0.0
    inventory_std = float(np.std(inventory_values)) if inventory_values.size else 0.0
    final_inventory = float(inventories[-1]) if inventories else 0.0
    quoted_steps = sum(bool(value) for value in quoted) if quoted is not None else 0
    number_of_steps = len(rewards)
    quote_rate = quoted_steps / number_of_steps if number_of_steps else 0.0
    action_counts = {action: 0 for action in range(6)}
    if actions is not None:
        for action in actions:
            action_counts[int(action)] += 1
    bid_fill_count = sum(bool(value) for value in bid_fills) if bid_fills is not None else 0
    ask_fill_count = sum(bool(value) for value in ask_fills) if ask_fills is not None else 0
    total_fills = bid_fill_count + ask_fill_count
    fill_rate = total_fills / number_of_steps if number_of_steps else 0.0

    return EpisodeMetrics(
        total_pnl=total_pnl,
        total_reward=total_reward,
        max_abs_inventory=max_abs_inventory,
        mean_abs_inventory=mean_abs_inventory,
        inventory_std=inventory_std,
        final_inventory=final_inventory,
        number_of_steps=number_of_steps,
        quoted_steps=quoted_steps,
        quote_rate=quote_rate,
        action_0_no_quote=action_counts[0],
        action_1_narrow=action_counts[1],
        action_2_medium=action_counts[2],
        action_3_wide=action_counts[3],
        action_4_skew_sell=action_counts[4],
        action_5_skew_buy=action_counts[5],
        bid_fills=bid_fill_count,
        ask_fills=ask_fill_count,
        total_fills=total_fills,
        fill_rate=fill_rate,
    )


def aggregate_episode_metrics(metrics: Sequence[EpisodeMetrics]) -> AggregateMetrics:
    """Compute mean and population std for a collection of episode metrics."""

    if not metrics:
        raise ValueError("At least one episode metric is required for aggregation.")

    summaries = {}
    for field in fields(EpisodeMetrics):
        values = np.array([float(getattr(metric, field.name)) for metric in metrics])
        summaries[field.name] = MetricSummary(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
        )

    return AggregateMetrics(**summaries)
