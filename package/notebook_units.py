"""
notebook_units.py

Small notebook-facing units shared by the LILAC SH5 notebooks.
The notebooks stay as the user execution entrypoints; this module only keeps
repeated setup code in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from constants import CONTROL_HZ
from controller import FileLanguageCommandSource, LILACSharedAutonomyController
from controller import load_latent_alignment
from data import DatasetRecorder, normalize_episode_type
from language import CanonicalLanguageDataset, CanonicalLanguageIndex
from paths import (
    CANONICAL_LANGUAGE_DATASET,
    DATA_DIR,
    RUNTIME_LANGUAGE_COMMAND,
    RUN_DIR,
    SCENE_XML,
)


@dataclass
class LILACRuntime:
    controller: LILACSharedAutonomyController
    command_source: FileLanguageCommandSource
    language_dataset: CanonicalLanguageDataset
    model: object | None
    language_index: CanonicalLanguageIndex | None
    use_lilac_model: bool
    run_dir: Path | None = None
    model_config: dict | None = None
    command_path: Path | None = None


def build_lilac_runtime(
        instruction = "",
        language_dataset_path = CANONICAL_LANGUAGE_DATASET,
        run_dir = RUN_DIR,
        command_path = RUNTIME_LANGUAGE_COMMAND,
        use_model_if_available = False,
        direct_mode_label      = "direct Vader5 IK",
        action_pos_scale       = 0.01,
        action_rot_scale       = 0.01,
    ):
    """
    Build the shared LILAC language/controller objects used by notebooks.
    """
    language_dataset = CanonicalLanguageDataset.load(language_dataset_path)

    if use_model_if_available:
        from lilac_model import LILACModel

        model, model_config = LILACModel.load_bundle(run_dir)
        language_index = CanonicalLanguageIndex.load(Path(run_dir) / "language_index.npz")
        print("[LILAC] loaded model from", run_dir)
    else:
        model = None
        model_config = None
        language_index = None
        print("[LILAC] collection mode:", direct_mode_label)

    controller = LILACSharedAutonomyController(
        model            = model,
        language_index   = language_index,
        language_dataset = language_dataset,
        action_pos_scale = action_pos_scale,
        action_rot_scale = action_rot_scale,
    )
    if str(instruction or "").strip():
        controller.set_instruction(instruction)

    command_source = FileLanguageCommandSource(command_path)
    runtime = LILACRuntime(
        controller       = controller,
        command_source   = command_source,
        language_dataset = language_dataset,
        model            = model,
        language_index   = language_index,
        use_lilac_model  = bool(use_model_if_available),
        run_dir          = Path(run_dir),
        model_config     = model_config,
        command_path      = Path(command_path),
    )
    return runtime


def write_runtime_language_command_header(command_path=RUNTIME_LANGUAGE_COMMAND):
    command_path = Path(command_path)
    command_path.parent.mkdir(parents=True, exist_ok=True)
    command_path.write_text(
        "# Runtime language commands for LILAC inference.\n"
        "# Recommended terminal input:\n"
        "#   /opt/anaconda3/envs/ri_motion_v5_env/bin/python project/LILAC_project/scripts/runtime_language_cli.py\n"
        "# Type any utterance as plain text. Exact canonical matches bypass Gemini.\n"
        "# Novel text is canonicalized by Gemini, then kind decides instruction vs correction.\n"
        "# Explicit commands also work: utterance <text>, instruction <text>, push <text>, pop, clear.\n"
        "\n",
        encoding="utf-8",
    )
    return command_path


def load_lilac_inference_runtime(
        action_pos_scale = 0.01,
        action_rot_scale = 0.017,
        run_dir          = RUN_DIR,
        command_path     = RUNTIME_LANGUAGE_COMMAND,
    ):
    """
    Load the single trained LILAC model and prepare an empty language stack.
    """
    run_dir = Path(run_dir)
    if not (run_dir / "model.pt").exists():
        raise FileNotFoundError("No model.pt in %s. Run 02_lilac_train_sh5.ipynb first." % run_dir)

    from lilac_model import LILACModel

    model, model_config = LILACModel.load_bundle(run_dir)
    language_dataset = CanonicalLanguageDataset.load(CANONICAL_LANGUAGE_DATASET)
    language_index = CanonicalLanguageIndex.load(run_dir / "language_index.npz")

    latent_alignment = None
    latent_alignment_path = run_dir / "latent_alignment.npz"
    if latent_alignment_path.exists():
        latent_alignment = load_latent_alignment(latent_alignment_path)
        print("[LILAC inference] loaded latent alignment", latent_alignment_path)
    else:
        print("[LILAC inference] no latent alignment found; using raw joystick z")

    controller = LILACSharedAutonomyController(
        model            = model,
        language_index   = language_index,
        language_dataset = language_dataset,
        latent_alignment = latent_alignment,
        action_pos_scale = action_pos_scale,
        action_rot_scale = action_rot_scale,
    )
    command_path = write_runtime_language_command_header(command_path)
    command_source = FileLanguageCommandSource(command_path)

    runtime = LILACRuntime(
        controller       = controller,
        command_source   = command_source,
        language_dataset = language_dataset,
        model            = model,
        language_index   = language_index,
        use_lilac_model  = True,
        run_dir          = run_dir,
        model_config     = model_config,
        command_path      = command_path,
    )
    print("[LILAC inference] loaded", run_dir)
    print(
        "state_dim",
        model_config["state_dim"],
        "action_dim",
        model_config["action_dim"],
        "latent_dim",
        model_config["latent_dim"],
    )
    return runtime


def append_runtime_language_command(command_path, command):
    """
    Append one runtime language command to the file-backed command queue.
    """
    line = str(command).strip()
    if not line:
        raise ValueError("Runtime language command cannot be empty.")

    path = Path(command_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    return line


def format_runtime_language_command(command, text=""):
    command = str(command).strip().lower()
    text = str(text).strip()

    if command in {"utterance", "say", "language"}:
        if not text:
            raise ValueError("utterance requires text.")
        return "utterance " + text
    if command in {"push", "correction"}:
        if not text:
            raise ValueError("push requires an utterance.")
        return "push " + text
    if command in {"instruction", "set"}:
        if not text:
            raise ValueError("instruction requires an utterance.")
        return "instruction " + text
    if command in {"pop", "clear"}:
        return command
    raise ValueError("Unsupported runtime language command: %s" % command)


def make_lilac_info(controller, source="init", z=None, action=None, alpha=1.0):
    """
    Create a compact info dictionary for recording and viewer overlays.
    """
    if z is None:
        z = np.zeros(2)
    if action is None:
        action = np.zeros(6)
    return {
        "source"         : source,
        "utterance"      : controller.active_utterance(),
        "alpha"          : float(alpha),
        "action"         : np.asarray(action, dtype=np.float64),
        "z"              : np.asarray(z, dtype=np.float64),
        "language_stack" : controller.language_stack.as_dict(),
    }


def make_collection_recorder(
        data_dir,
        task_name,
        instruction,
        episode_type,
        joint_names,
        object_state,
        hz = CONTROL_HZ,
    ):
    """
    Create and start a recorder for a LILAC collection notebook.
    """
    return DatasetRecorder(
        data_dir     = data_dir,
        task         = task_name,
        instruction  = instruction,
        episode_type = episode_type,
        hz           = hz,
        joint_names  = joint_names,
        object_state = object_state,
    ).start()


def collection_episode_dir(data_dir, task_name, episode_type):
    return Path(data_dir) / str(task_name) / normalize_episode_type(episode_type)


def count_collection_episodes(episode_dir):
    return len(list(Path(episode_dir).glob("*.npz")))


def trim_overlay_text(value, max_len=58):
    value = str(value)
    if len(value) <= int(max_len):
        return value
    return value[:int(max_len) - 3] + "..."


def make_collection_overlay_rows(
        data_dir,
        task_name,
        episode_type,
        instruction,
        initial_episode_count,
        saved_recordings,
        record_on,
        recorder,
    ):
    episode_dir = collection_episode_dir(data_dir, task_name, episode_type)
    episode_type = normalize_episode_type(episode_type)
    session_count = len(saved_recordings or [])
    total_count = int(initial_episode_count) + session_count
    n_frames = 0 if recorder is None else len(recorder.frames)
    return [
        ("Task", trim_overlay_text(task_name, 36)),
        ("Episode", trim_overlay_text(episode_type, 36)),
        ("Dataset", trim_overlay_text(episode_dir, 58)),
        ("Saved", "%d total (+%d session)" % (total_count, session_count)),
        ("Record", "ON (%d frames)" % n_frames if record_on else "OFF"),
        ("Instruction", trim_overlay_text(instruction, 58)),
    ]


def save_collection_recorder(recorder, saved_recordings=None):
    """
    Save a recorder if it has data. Returns the save result or None.
    """
    if recorder is None or len(recorder.frames) <= 1:
        return None
    saved = recorder.save()
    if saved_recordings is not None:
        saved_recordings.append(saved)
    print("[dataset] saved", saved["npz_path"])
    return saved


def cancel_collection_recorder(recorder):
    """
    Discard an in-progress recording without saving an episode file.
    """
    n_frames = 0 if recorder is None else len(recorder.frames)
    print("[dataset] canceled recording; discarded %d frames" % n_frames)
    return n_frames
