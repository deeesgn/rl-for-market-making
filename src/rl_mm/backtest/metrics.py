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
    final_inventory: float
    number_of_steps: int

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
    final_inventory: MetricSummary
    number_of_steps: MetricSummary

    def as_dict(self) -> dict[str, MetricSummary]:
        return asdict(self)


def compute_episode_metrics(
    *,
    rewards: Sequence[float],
    pnls: Sequence[float],
    inventories: Sequence[float],
) -> EpisodeMetrics:
    """Compute comparison metrics from one completed episode."""

    total_pnl = float(pnls[-1]) if pnls else 0.0
    total_reward = float(sum(rewards))
    max_abs_inventory = max((abs(float(value)) for value in inventories), default=0.0)
    final_inventory = float(inventories[-1]) if inventories else 0.0

    return EpisodeMetrics(
        total_pnl=total_pnl,
        total_reward=total_reward,
        max_abs_inventory=max_abs_inventory,
        final_inventory=final_inventory,
        number_of_steps=len(rewards),
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
