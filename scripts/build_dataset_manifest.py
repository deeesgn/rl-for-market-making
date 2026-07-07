"""Build a JSON manifest for processed parquet datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_mm.data.manifest import write_dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed dataset manifest.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output_path = write_dataset_manifest(args.input_dir, args.output)
    print(f"wrote dataset manifest: {output_path}")


if __name__ == "__main__":
    main()
