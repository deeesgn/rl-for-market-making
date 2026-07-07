from pathlib import Path

from rl_mm.config import load_experiment_protocol
from rl_mm.data.splits import compute_chronological_splits, format_split_ranges


def test_compute_chronological_splits_are_ordered_and_complete() -> None:
    protocol = load_experiment_protocol(Path("configs/experiment_protocol.yaml"))

    train, validation, test = compute_chronological_splits(protocol)

    assert train.name == "train"
    assert validation.name == "validation"
    assert test.name == "test"
    assert train.start == protocol.start_date
    assert train.end == validation.start
    assert validation.end == test.start
    assert test.end == protocol.end_date
    assert train.days + validation.days + test.days == (
        protocol.end_date - protocol.start_date
    ).days
    assert train.start < train.end < validation.end < test.end


def test_format_split_ranges() -> None:
    protocol = load_experiment_protocol(Path("configs/experiment_protocol.yaml"))
    output = format_split_ranges(compute_chronological_splits(protocol))

    assert "inclusive start, exclusive end" in output
    assert "- train:" in output
    assert "- validation:" in output
    assert "- test:" in output
