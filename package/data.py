"""
data.py

Dataset acquisition and preprocessing utilities for LILAC on the SH5 right arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
import uuid

import numpy as np

from constants import (
    ACTION_SPACE,
    CONTROL_HZ,
    LATENT_DIM,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_FINGER_JOINT_NAMES,
)
from language import CanonicalLanguageDataset, DatasetAlphaLabeler


EPISODE_TYPE_ALIASES = {
    "instruction": "instruction",
    "full_task": "instruction",
    "full task": "instruction",
    "instrction": "instruction",
    "correction": "correction",
}


def normalize_episode_type(episode_type):
    """
    Normalize old collection folder names to the current two training splits.
    """
    key = str(episode_type or "").strip().lower()
    return EPISODE_TYPE_ALIASES.get(key, key)


def wrap_to_pi(x):
    return (np.asarray(x, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def T_to_pose6(T):
    """
    Convert a homogeneous transform to [x, y, z, roll, pitch, yaw].
    """
    from ri_motion_v5_package.kinematics.transforms import r2rpy, t2p, t2r

    T = np.asarray(T, dtype=np.float64)
    p = t2p(T)
    rpy = r2rpy(t2r(T), unit="rad")
    return np.concatenate([p, rpy]).astype(np.float64)


def pose6_delta(pose6_prev, pose6_next):
    """
    Compute paper action: Cartesian delta and Euler-angle delta.
    """
    pose6_prev = np.asarray(pose6_prev, dtype=np.float64).reshape(6,)
    pose6_next = np.asarray(pose6_next, dtype=np.float64).reshape(6,)
    dp = pose6_next[:3] - pose6_prev[:3]
    drpy = wrap_to_pi(pose6_next[3:] - pose6_prev[3:])
    return np.concatenate([dp, drpy]).astype(np.float64)


def make_dataset_timestamp(timestamp=None):
    """
    Return dataset timestamps for metadata and filename-safe episode ids.
    """
    timestamp = time.time() if timestamp is None else float(timestamp)
    local_time = time.localtime(timestamp)
    return {
        "created_at": time.strftime("%m/%d %H:%M", local_time),
        "created_at_safe": time.strftime("%m%d_%H%M", local_time),
        "created_at_unix": timestamp,
    }


def make_episode_id(prefix="episode", stamp=None):
    stamp = time.strftime("%m%d_%H%M") if stamp is None else str(stamp)
    return "%s_%s_%s" % (prefix, stamp, uuid.uuid4().hex[:8])


def json_dumps_safe(obj):
    def default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, (np.bool_,)):
            return bool(x)
        return str(x)

    return json.dumps(obj, default=default, ensure_ascii=True)


@dataclass
class DatasetRecorder:
    """
    Fixed-rate friendly recorder for full-task and correction demonstrations.
    """

    data_dir: Path | str = Path("data")
    task: str = "task"
    instruction: str = ""
    episode_type: str = "instruction"
    hz: float = CONTROL_HZ
    joint_names: list[str] = field(default_factory=lambda: list(RIGHT_ARM_JOINT_NAMES))
    hand_joint_names: list[str] = field(default_factory=lambda: list(RIGHT_FINGER_JOINT_NAMES))
    object_state: np.ndarray | None = None
    episode_id: str | None = None

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.episode_type = normalize_episode_type(self.episode_type)
        timestamp_meta = make_dataset_timestamp()
        self.created_at = timestamp_meta["created_at"]
        self.created_at_safe = timestamp_meta["created_at_safe"]
        self.created_at_unix = timestamp_meta["created_at_unix"]
        if self.episode_id is None:
            self.episode_id = make_episode_id(self.task, self.created_at_safe)
        self.frames = []
        self.time_start = None

    def start(self):
        self.time_start = time.time()
        return self

    def append(
            self,
            q_arm,
            T_ee,
            action_ee_delta  = None,
            latent_z         = None,
            q_hand           = None,
            q_all            = None,
            vader5_state     = None,
            object_state     = None,
            correction_stack = None,
            active_utterance = None,
            alpha            = None,
            right_grasp      = None,
            ik_err           = None,
            contact_on       = None,
            extra            = None,
            timestamp        = None,
        ):
        if self.time_start is None:
            self.start()

        timestamp = time.time() if timestamp is None else float(timestamp)
        T_ee = np.asarray(T_ee, dtype=np.float64)
        pose6 = T_to_pose6(T_ee)

        frame = {
            "timestamp": timestamp,
            "elapsed": timestamp - self.time_start,
            "q_arm": np.asarray(q_arm, dtype=np.float64).reshape(-1,),
            "q_hand": self._optional_array(q_hand),
            "q_all": self._optional_array(q_all),
            "T_ee": T_ee.reshape(4, 4),
            "ee_pose": pose6,
            "action_ee_delta": self._optional_array(action_ee_delta, width=6),
            "latent_z": self._optional_array(latent_z),
            "vader5_state": vader5_state,
            "object_state": self._optional_array(object_state),
            "correction_stack": correction_stack,
            "active_utterance": active_utterance,
            "alpha": np.nan if alpha is None else float(alpha),
            "right_grasp": np.nan if right_grasp is None else float(right_grasp),
            "ik_err": np.nan if ik_err is None else float(ik_err),
            "contact_on": False if contact_on is None else bool(contact_on),
            "extra": extra if extra is not None else {},
        }
        self.frames.append(frame)

    @staticmethod
    def _optional_array(value, width=None):
        if value is None:
            return None
        value = np.asarray(value, dtype=np.float64).reshape(-1,)
        if width is not None:
            value = value.reshape(width,)
        return value

    def save(self):
        if len(self.frames) == 0:
            raise ValueError("Cannot save an empty episode.")

        out_dir = self.data_dir / self.task / self.episode_type
        out_dir.mkdir(parents=True, exist_ok=True)
        npz_path = out_dir / ("%s.npz" % self.episode_id)
        json_path = out_dir / ("%s.json" % self.episode_id)

        arrays = {
            "timestamp": np.asarray([f["timestamp"] for f in self.frames], dtype=np.float64),
            "elapsed": np.asarray([f["elapsed"] for f in self.frames], dtype=np.float64),
            "created_at": np.asarray([self.created_at], dtype=object),
            "created_at_safe": np.asarray([self.created_at_safe], dtype=object),
            "created_at_unix": np.asarray([self.created_at_unix], dtype=np.float64),
            "q_arm": np.stack([f["q_arm"] for f in self.frames], axis=0),
            "T_ee": np.stack([f["T_ee"] for f in self.frames], axis=0),
            "ee_pose": np.stack([f["ee_pose"] for f in self.frames], axis=0),
            "alpha": np.asarray([f["alpha"] for f in self.frames], dtype=np.float64),
            "right_grasp": np.asarray([f["right_grasp"] for f in self.frames], dtype=np.float64),
            "ik_err": np.asarray([f["ik_err"] for f in self.frames], dtype=np.float64),
            "contact_on": np.asarray([f["contact_on"] for f in self.frames], dtype=np.bool_),
            "active_utterance": np.asarray([str(f["active_utterance"] or "") for f in self.frames], dtype=object),
            "vader5_state_json": np.asarray([json_dumps_safe(f["vader5_state"]) for f in self.frames], dtype=object),
            "correction_stack_json": np.asarray(
                [json_dumps_safe(f["correction_stack"]) for f in self.frames],
                dtype=object,
            ),
            "extra_json": np.asarray([json_dumps_safe(f["extra"]) for f in self.frames], dtype=object),
        }

        arrays["q_hand"] = self._stack_optional("q_hand")
        arrays["q_all"] = self._stack_optional("q_all")
        arrays["action_ee_delta"] = self._stack_optional("action_ee_delta", width=6)
        arrays["latent_z"] = self._stack_optional("latent_z")
        object_state_array = self._stack_object_state()
        if object_state_array is not None:
            arrays["object_state"] = object_state_array
        elif self.object_state is not None:
            arrays["object_state"] = np.asarray(self.object_state, dtype=np.float64).reshape(-1,)

        np.savez_compressed(npz_path, **arrays)

        meta = {
            "episode_id": self.episode_id,
            "created_at": self.created_at,
            "created_at_safe": self.created_at_safe,
            "created_at_unix": self.created_at_unix,
            "task": self.task,
            "instruction": self.instruction,
            "episode_type": self.episode_type,
            "hz": float(self.hz),
            "action_space": ACTION_SPACE,
            "joint_names": list(self.joint_names),
            "hand_joint_names": list(self.hand_joint_names),
            "n_frames": len(self.frames),
            "npz": str(npz_path),
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)

        return {
            "npz_path": npz_path,
            "json_path": json_path,
            "meta": meta,
        }

    def _stack_optional(self, key, width=None):
        values = [f[key] for f in self.frames]
        valid = [v for v in values if v is not None]
        if not valid:
            width = 0 if width is None else int(width)
            return np.full((len(values), width), np.nan, dtype=np.float64)

        width = len(valid[0]) if width is None else int(width)
        out = np.full((len(values), width), np.nan, dtype=np.float64)
        for idx, value in enumerate(values):
            if value is None:
                continue
            value = np.asarray(value, dtype=np.float64).reshape(-1,)
            out[idx, :min(width, len(value))] = value[:width]
        return out

    def _stack_object_state(self):
        values = [f["object_state"] for f in self.frames]
        valid = [v for v in values if v is not None]
        if not valid:
            return None

        width = len(valid[0])
        out = np.full((len(values), width), np.nan, dtype=np.float64)
        for idx, value in enumerate(values):
            if value is None:
                continue
            value = np.asarray(value, dtype=np.float64).reshape(-1,)
            out[idx, :min(width, len(value))] = value[:width]
        return out


def iter_episode_npzs(data_dir, tasks=None, episode_types=None):
    data_dir = Path(data_dir)
    tasks = None if tasks is None else set(tasks)
    episode_types = (
        None
        if episode_types is None
        else {normalize_episode_type(episode_type) for episode_type in episode_types}
    )

    for npz_path in sorted(data_dir.glob("*/*/*.npz")):
        task = npz_path.parents[1].name
        episode_type = normalize_episode_type(npz_path.parent.name)
        if tasks is not None and task not in tasks:
            continue
        if episode_types is not None and episode_type not in episode_types:
            continue
        yield npz_path


def load_episode(npz_path):
    npz_path = Path(npz_path)
    meta_path = npz_path.with_suffix(".json")
    data = np.load(npz_path, allow_pickle=True)
    meta = {}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    return data, meta


def resolve_episode_object_state(data, object_state=None):
    if object_state is not None:
        return np.asarray(object_state, dtype=np.float64).reshape(-1,)
    if "object_state" not in data:
        return np.zeros((0,), dtype=np.float64)

    obj = np.asarray(data["object_state"], dtype=np.float64)
    if obj.ndim >= 2:
        return obj
    return obj.reshape(-1,)


def get_frame_object_state(obj, idx):
    obj = np.asarray(obj, dtype=np.float64)
    if obj.ndim >= 2:
        return obj[idx].reshape(-1,)
    return obj.reshape(-1,)


def get_canonical_training_entry(utterance, language_dataset, canonical_cache, npz_path):
    if utterance not in canonical_cache:
        try:
            canonical_cache[utterance] = language_dataset.get(utterance)
        except KeyError as exc:
            raise ValueError(
                "Recorded training utterance is not in the canonical language dataset: "
                "%r in %s. Training data must store GT canonical utterance text or id; "
                "Gemini canonicalization is used only at deployment time."
                % (utterance, npz_path)
            ) from exc
    return canonical_cache[utterance]


def resolve_training_utterance(active, instruction, episode_type, language_dataset, canonical_cache, npz_path, idx):
    """
    Choose the GT canonical utterance for a frame.

    Correction-only collection can store the correction in episode metadata even
    when the runtime stack was not pushed before recording. In that case, prefer
    the metadata instruction if it is a canonical correction.
    """
    instruction = str(instruction or "")
    episode_type = normalize_episode_type(episode_type)

    if instruction:
        entry = get_canonical_training_entry(
            utterance        = instruction,
            language_dataset = language_dataset,
            canonical_cache  = canonical_cache,
            npz_path         = npz_path,
        )
        if episode_type == "correction" and entry.kind == "correction":
            return entry
        if episode_type == "instruction" and entry.kind == "instruction":
            return entry

    utterance = ""
    if active is not None:
        utterance = str(active[idx])
    if not utterance:
        utterance = instruction
    return get_canonical_training_entry(
        utterance        = utterance,
        language_dataset = language_dataset,
        canonical_cache  = canonical_cache,
        npz_path         = npz_path,
    )


def build_training_arrays(
        episode_npzs,
        object_state       = None,
        alpha_labeler      = None,
        language_dataset   = None,
        normalize_actions  = True,
    ):
    """
    Build LILAC arrays from recorded episodes.

    State = q_arm(7) + ee_pose(6) + optional flattened object_state.
    Action = delta in ee_pose between consecutive frames.
    """
    if language_dataset is None:
        language_dataset = CanonicalLanguageDataset.load()
    if alpha_labeler is None:
        alpha_labeler = DatasetAlphaLabeler(language_dataset)
    if object_state is not None:
        object_state = np.asarray(object_state, dtype=np.float64).reshape(-1,)

    states = []
    actions = []
    utterances = []
    alphas = []
    latent_zs = []
    episode_ids = []
    canonical_cache = {}

    for npz_path in episode_npzs:
        data, meta = load_episode(npz_path)
        q_arm = np.asarray(data["q_arm"], dtype=np.float64)
        ee_pose = np.asarray(data["ee_pose"], dtype=np.float64)
        if len(q_arm) < 2:
            continue

        obj = resolve_episode_object_state(data, object_state=object_state)
        recorded_latent_z = np.asarray(data["latent_z"], dtype=np.float64) if "latent_z" in data else None
        active = data["active_utterance"] if "active_utterance" in data else None
        instruction = meta.get("instruction", "")
        episode_type = meta.get("episode_type", "")
        episode_id = meta.get("episode_id", Path(npz_path).stem)

        for idx in range(len(q_arm) - 1):
            entry = resolve_training_utterance(
                active           = active,
                instruction      = instruction,
                episode_type     = episode_type,
                language_dataset = language_dataset,
                canonical_cache  = canonical_cache,
                npz_path         = npz_path,
                idx              = idx,
            )
            utterance = entry.text
            alpha = float(alpha_labeler(entry.id))

            obj_i = get_frame_object_state(obj, idx)
            state = np.concatenate([q_arm[idx], ee_pose[idx], obj_i]).astype(np.float32)
            action = pose6_delta(ee_pose[idx], ee_pose[idx + 1]).astype(np.float32)
            if normalize_actions:
                norm = float(np.linalg.norm(action))
                if norm > 1e-12:
                    action = action / norm

            if recorded_latent_z is not None and idx < len(recorded_latent_z):
                latent_z = np.asarray(recorded_latent_z[idx], dtype=np.float64).reshape(-1,)[:LATENT_DIM]
            else:
                latent_z = np.full((LATENT_DIM,), np.nan, dtype=np.float64)

            states.append(state)
            actions.append(action)
            utterances.append(utterance)
            alphas.append(alpha)
            latent_zs.append(latent_z.astype(np.float32))
            episode_ids.append(episode_id)

    if not states:
        raise ValueError("No training samples were produced.")

    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "utterances": np.asarray(utterances, dtype=object),
        "alphas": np.asarray(alphas, dtype=np.float32),
        "latent_z": np.asarray(latent_zs, dtype=np.float32),
        "episode_ids": np.asarray(episode_ids, dtype=object),
    }


def save_training_arrays(arrays, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return out_path


def sample_correction_start_indices(n_frames, n_samples, min_margin=5, rng=None):
    """
    Paper-style helper for sampling intermediate replay states for corrections.
    """
    rng = np.random.default_rng() if rng is None else rng
    lo = int(min_margin)
    hi = max(lo + 1, int(n_frames) - int(min_margin))
    n_samples = min(int(n_samples), max(0, hi - lo))
    if n_samples <= 0:
        return np.asarray([], dtype=np.int64)
    return np.sort(rng.choice(np.arange(lo, hi), size=n_samples, replace=False))
