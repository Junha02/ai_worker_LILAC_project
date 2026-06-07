from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def project_root() -> Path:
    configured = os.environ.get("LILAC_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[4]


def add_core_package_to_path() -> Path:
    core_dir = project_root() / "package"
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))
    return core_dir


def canonical_dataset_path() -> Path:
    return project_root() / "data" / "language" / "lilac_canonical_utterances.json"


def run_dir() -> Path:
    return project_root() / "runs" / "lilac_sh5_right"


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def stamp(msg, node, frame_id="world"):
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = frame_id
    return msg
