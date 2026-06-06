#!/usr/bin/env python3
"""
Append runtime language commands for LILAC inference from a terminal.
"""

from pathlib import Path
import argparse
import os
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_DIR / "package"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.append(str(PACKAGE_DIR))

from language import CanonicalLanguageDataset, load_project_env
from notebook_units import append_runtime_language_command, format_runtime_language_command
from paths import CANONICAL_LANGUAGE_DATASET, RUNTIME_LANGUAGE_COMMAND


COMMAND_WORDS = {"utterance", "say", "language", "push", "correction", "instruction", "set", "pop", "clear"}
QUIT_WORDS = {"q", "quit", "exit"}


def has_gemini_key():
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send text commands to the file-backed LILAC runtime language queue.",
    )
    parser.add_argument(
        "command",
        nargs="*",
        help=(
            "Optional one-shot command. Examples: push pour water | pop | pour water. "
            "Plain text is treated as utterance <text>."
        ),
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=RUNTIME_LANGUAGE_COMMAND,
        help="Runtime command file polled by the inference notebook.",
    )
    parser.add_argument(
        "--language-dataset",
        type=Path,
        default=CANONICAL_LANGUAGE_DATASET,
    )
    return parser.parse_args()


def line_to_command(line, command_path=None):
    line = str(line).strip()
    if not line:
        return None

    command, _, payload = line.partition(" ")
    command = command.strip().lower()
    payload = payload.strip()

    if command in QUIT_WORDS:
        return "quit"
    if command in COMMAND_WORDS:
        return format_runtime_language_command(command, payload)
    return format_runtime_language_command("utterance", line)


def print_candidates(path):
    try:
        dataset = CanonicalLanguageDataset.load(path)
    except Exception as exc:
        print("[language-cli] could not load canonical dataset:", exc)
        return

    print("[language-cli] canonical candidates:")
    for entry in dataset.entries:
        print("  - %-12s %-11s %s" % (entry.id, entry.kind, entry.text))


def send_line(command_path, raw_line):
    command_line = line_to_command(raw_line, command_path=command_path)
    if command_line is None:
        return None
    if command_line == "quit":
        return "quit"
    appended = append_runtime_language_command(command_path, command_line)
    print("[language-cli] queued:", appended)
    return appended


def interactive_loop(args):
    load_project_env()
    print("[language-cli] writing to:", args.path)
    print("[language-cli] Gemini key:", "loaded" if has_gemini_key() else "missing")
    print_candidates(args.language_dataset)
    print("")
    print("Type any utterance as plain text. Exact canonical matches bypass Gemini.")
    print("Novel text uses Gemini, then the selected canonical kind decides instruction vs correction.")
    print("Explicit commands also work: utterance <text>, instruction <text>, push <text>, pop, clear.")
    print("Press Ctrl-D or type quit to exit.")
    while True:
        try:
            raw = input("lilac> ")
        except EOFError:
            print("")
            break

        try:
            result = send_line(args.path, raw)
            if result == "quit":
                break
        except Exception as exc:
            print("[language-cli] error:", exc)


def main():
    load_project_env()
    args = parse_args()
    if args.command:
        raw = " ".join(args.command)
        result = send_line(args.path, raw)
        return 0 if result else 1

    interactive_loop(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
