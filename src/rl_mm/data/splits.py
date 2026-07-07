"""Chronological train/validation/test split utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from rl_mm.config import ExperimentProtocol


@dataclass(frozen=True)
class DateRange:
    """Inclusive start and exclusive end date range."""

    name: str
    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days


def compute_chronological_splits(protocol: ExperimentProtocol) -> tuple[DateRange, ...]:
    """Compute strict chronological split ranges from the experiment protocol."""

    total_days = (protocol.end_date - protocol.start_date).days
    if total_days <= 0:
        raise ValueError("Protocol date range must contain at least one day.")

    train_days = round(total_days * protocol.train_split)
    validation_days = round(total_days * protocol.validation_split)
    test_days = total_days - train_days - validation_days
    if min(train_days, validation_days, test_days) <= 0:
        raise ValueError("All chronological splits must contain at least one day.")

    train_start = protocol.start_date
    train_end = train_start + timedelta(days=train_days)
    validation_end = train_end + timedelta(days=validation_days)

    return (
        DateRange("train", train_start, train_end),
        DateRange("validation", train_end, validation_end),
        DateRange("test", validation_end, protocol.end_date),
    )


def format_split_ranges(splits: tuple[DateRange, ...]) -> str:
    lines = ["Chronological data splits (inclusive start, exclusive end):"]
    for split in splits:
        lines.append(f"- {split.name}: {split.start.isoformat()} -> {split.end.isoformat()}")
    return "\n".join(lines)
