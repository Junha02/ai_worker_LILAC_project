#!/usr/bin/env python3
"""
Prepare LILAC training arrays from recorded SH5 right-arm episodes.
"""

from pathlib import Path
import argparse
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_DIR / "package"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.append(str(PACKAGE_DIR))

from training import prepare_training_arrays
from paths import CANONICAL_LANGUAGE_DATASET, DATA_DIR, TRAINING_ARRAYS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=TRAINING_ARRAYS,
    )
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--episode-types", nargs="*", default=None)
    parser.add_argument(
        "--language-dataset",
        type=Path,
        default=CANONICAL_LANGUAGE_DATASET,
    )
    parser.add_argument("--no-normalize-actions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episode_types is None:
        args.episode_types = ["instruction", "correction"]
    out = prepare_training_arrays(
        data_dir              = args.data_dir,
        out_path              = args.out,
        tasks                 = args.tasks,
        episode_types         = args.episode_types,
        language_dataset_path = args.language_dataset,
        normalize_actions     = not args.no_normalize_actions,
    )
    print("Saved training arrays:", out)


if __name__ == "__main__":
    main()
