#!/usr/bin/env python3
"""
Train the LILAC latent-action model from prepared arrays.
"""

from pathlib import Path
import argparse
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_DIR / "package"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.append(str(PACKAGE_DIR))

from training import train_lilac_from_arrays
from paths import CANONICAL_LANGUAGE_DATASET, RUN_DIR, TRAINING_ARRAYS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arrays",
        type=Path,
        default=TRAINING_ARRAYS,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=RUN_DIR,
    )
    parser.add_argument("--language-index", type=Path, default=None)
    parser.add_argument(
        "--language-dataset",
        type=Path,
        default=CANONICAL_LANGUAGE_DATASET,
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=21)
    return parser.parse_args()


def main():
    args = parse_args()
    result = train_lilac_from_arrays(
        arrays_path           = args.arrays,
        run_dir               = args.run_dir,
        language_index_path   = args.language_index,
        language_dataset_path = args.language_dataset,
        batch_size            = args.batch_size,
        n_epochs              = args.epochs,
        lr                    = args.lr,
        weight_decay          = args.weight_decay,
        seed                  = args.seed,
        overwrite_run         = True,
        prune_existing_runs   = True,
    )
    print("Saved run:", result["run_dir"])
    print("Removed old runs:", [str(path) for path in result.get("removed_run_dirs", [])])


if __name__ == "__main__":
    main()
