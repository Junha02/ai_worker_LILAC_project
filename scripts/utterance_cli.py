#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_DIR / "package"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from paths import RUNTIME_UTTERANCE_FILE


COMMANDS = {"utterance", "instruction", "set", "push", "correction", "pop", "clear"}


def normalize_line(text):
    line = text.strip()
    if not line:
        return ""
    head, _, payload = line.partition(" ")
    if head.lower() in COMMANDS:
        return line
    return "utterance " + line


def append_request(line):
    RUNTIME_UTTERANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RUNTIME_UTTERANCE_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    print("[LILAC_ROS language-cli] type text, pop, clear, or Ctrl-D to exit")
    while True:
        try:
            raw = input("lilac_ros> ")
        except EOFError:
            print()
            return
        line = normalize_line(raw)
        if not line:
            continue
        append_request(line)
        print("[language-cli] queued:", line)


if __name__ == "__main__":
    main()
