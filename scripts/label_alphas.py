#!/usr/bin/env python3
"""
Export alpha labels from the canonical LILAC language dataset.
"""

from pathlib import Path
import argparse
import json
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_DIR / "package"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.append(str(PACKAGE_DIR))

from language import CanonicalLanguageDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language-dataset",
        type=Path,
        default=PROJECT_DIR / "data" / "language" / "lilac_canonical_utterances.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_DIR / "data" / "canonical-alpha-labels.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = CanonicalLanguageDataset.load(args.language_dataset)
    payload = {entry.text: float(entry.alpha) for entry in dataset.entries}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    print("Saved canonical alpha labels:", args.out)


if __name__ == "__main__":
    main()
