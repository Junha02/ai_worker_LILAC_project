"""
paths.py

Single source of truth for LILAC project paths used by notebooks and scripts.
"""

from __future__ import annotations

from pathlib import Path


THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parent
PROJECT_DIR = PACKAGE_DIR.parent
LAB_DIR = PROJECT_DIR.parents[1]

DATA_DIR = PROJECT_DIR / "data"
LANGUAGE_DIR = DATA_DIR / "language"
TRAINING_DIR = DATA_DIR / "training"
RUNS_DIR = PROJECT_DIR / "runs"
CACHE_DIR = PROJECT_DIR / "cache"

CANONICAL_LANGUAGE_DATASET = LANGUAGE_DIR / "lilac_canonical_utterances.json"
CANONICAL_LANGUAGE_INDEX = LANGUAGE_DIR / "language_index.npz"
TRAINING_ARRAYS = TRAINING_DIR / "lilac_sh5_right_arrays.npz"
RUN_DIR = RUNS_DIR / "lilac_sh5_right"
RUNTIME_LANGUAGE_COMMAND = DATA_DIR / "runtime_language_command.txt"

SCENE_XML = str(PROJECT_DIR / "notebook" / "xml" / "scene_ffw_sh5_lilac.xml")
