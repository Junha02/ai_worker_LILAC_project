#!/usr/bin/env python3
"""
Precompute SBERT language embeddings for the canonical LILAC utterance set.
"""

from pathlib import Path
import argparse
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_DIR / "package"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.append(str(PACKAGE_DIR))

from language import (
    CanonicalLanguageDataset,
    CanonicalLanguageIndex,
    TransformerLanguageEmbedder,
)


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
        default=PROJECT_DIR / "data" / "language" / "language_index.npz",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-xlm-r-multilingual-v1",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_DIR / "cache" / "sbert",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = CanonicalLanguageDataset.load(args.language_dataset)
    embedder = TransformerLanguageEmbedder(
        model_name=args.model_name,
        cache_dir=str(args.cache_dir),
        device=args.device,
    )
    index = CanonicalLanguageIndex.build(dataset=dataset, embedder=embedder)
    out = index.save(args.out)
    print("Saved canonical language index:", out)


if __name__ == "__main__":
    main()
