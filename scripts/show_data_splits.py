"""Show chronological train/validation/test date splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_mm.config import load_experiment_protocol
from rl_mm.data.splits import compute_chronological_splits, format_split_ranges


def main() -> None:
    parser = argparse.ArgumentParser(description="Show experiment data splits.")
    parser.add_argument("--protocol", type=Path, default=Path("configs/experiment_protocol.yaml"))
    args = parser.parse_args()

    protocol = load_experiment_protocol(args.protocol)
    splits = compute_chronological_splits(protocol)
    print(format_split_ranges(splits))


if __name__ == "__main__":
    main()
