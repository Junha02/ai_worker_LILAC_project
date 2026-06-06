"""
controller.py

Runtime LILAC shared-autonomy controller for SH5 right-palm target control.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from constants import ACTION_DIM, LATENT_DIM
from language import (
    CanonicalLanguageDataset,
    CanonicalLanguageIndex,
    DatasetAlphaLabeler,
    GeminiUtteranceSelector,
    LanguageStack,
)


def apply_ee_delta_to_T(
        T_curr,
        action,
        translation_frame="world",
        rotation_frame="local",
    ):
    """
    Apply a 6-DoF end-effector delta to a target transform.
    """
    from ri_motion_v5_package.kinematics.transforms import pr2t, rpy2r, t2p, t2r

    action = np.asarray(action, dtype=np.float64).reshape(6,)
    T_curr = np.asarray(T_curr, dtype=np.float64).reshape(4, 4)
    p_curr = t2p(T_curr).copy()
    R_curr = t2r(T_curr).copy()

    dp = action[:3]
    drpy = action[3:]

    if translation_frame == "world":
        p_next = p_curr + dp
    elif translation_frame == "local":
        p_next = p_curr + R_curr @ dp
    else:
        raise ValueError("Invalid translation_frame:[%s]" % translation_frame)

    R_delta = rpy2r(drpy, unit="rad")
    if rotation_frame == "local":
        R_next = R_curr @ R_delta
    elif rotation_frame == "world":
        R_next = R_delta @ R_curr
    else:
        raise ValueError("Invalid rotation_frame:[%s]" % rotation_frame)

    return pr2t(p_next, R_next)


def load_latent_alignment(path):
    payload = np.load(path, allow_pickle=True)
    return {
        "weight": np.asarray(payload["weight"], dtype=np.float64),
        "bias": np.asarray(payload["bias"], dtype=np.float64),
    }


def apply_latent_alignment(z, alignment, clip=True):
    z = np.asarray(z, dtype=np.float64).reshape(LATENT_DIM,)
    if alignment is None:
        return z.copy()
    weight = np.asarray(alignment["weight"], dtype=np.float64).reshape(LATENT_DIM, LATENT_DIM)
    bias = np.asarray(alignment.get("bias", np.zeros(LATENT_DIM)), dtype=np.float64).reshape(LATENT_DIM,)
    z_aligned = z @ weight + bias
    if clip:
        z_aligned = np.clip(z_aligned, -1.0, 1.0)
    return z_aligned.astype(np.float64)


class LILACSharedAutonomyController:
    """
    Shared autonomy runtime: canonical language stack + latent z -> 6-DoF EE delta.
    """

    def __init__(
            self,
            model=None,
            language_index=None,
            language_dataset=None,
            utterance_selector=None,
            alpha_labeler=None,
            latent_alignment=None,
            clip_aligned_latent=True,
            action_pos_scale=0.02,
            action_rot_scale=0.15,
        ):
        self.model = model
        self.language_dataset = language_dataset
        if self.language_dataset is None and language_index is not None:
            self.language_dataset = language_index.dataset
        if self.language_dataset is None:
            self.language_dataset = CanonicalLanguageDataset.load()

        self.language_index = language_index
        self.utterance_selector = utterance_selector
        if self.utterance_selector is None:
            self.utterance_selector = GeminiUtteranceSelector(self.language_dataset)
        self.alpha_labeler = (
            alpha_labeler
            if alpha_labeler is not None
            else DatasetAlphaLabeler(self.language_dataset)
        )
        self.latent_alignment = latent_alignment
        self.clip_aligned_latent = bool(clip_aligned_latent)
        self.action_pos_scale = float(action_pos_scale)
        self.action_rot_scale = float(action_rot_scale)
        self.language_stack = LanguageStack()
        self.last_error = None

    def canonicalize(self, utterance, kind=None):
        return self.utterance_selector.select(utterance, kind=kind)

    def apply_utterance(self, utterance):
        """
        Canonicalize free-form runtime text across all candidates.

        Exact canonical text/id/alias matches return locally. Only novel text
        reaches Gemini. The selected canonical kind decides stack behavior.
        """
        entry = self.canonicalize(utterance, kind=None)
        if entry.kind == "instruction":
            self.language_stack.set_instruction(entry.text)
            return entry, "instruction"
        if entry.kind == "correction":
            self.language_stack.push(entry.text)
            return entry, "push"
        raise ValueError("Unsupported canonical utterance kind: %s" % entry.kind)

    def set_instruction(self, instruction, canonicalize=True):
        if canonicalize:
            entry = self.canonicalize(instruction, kind="instruction")
        else:
            entry = self.language_dataset.get(instruction, kind="instruction")
        self.language_stack.set_instruction(entry.text)
        return entry

    def push_correction(self, correction, canonicalize=True):
        if canonicalize:
            entry = self.canonicalize(correction, kind="correction")
        else:
            entry = self.language_dataset.get(correction, kind="correction")
        self.language_stack.push(entry.text)
        return entry

    def pop_correction(self):
        return self.language_stack.pop()

    def active_utterance(self):
        return self.language_stack.active()

    def active_entry(self):
        if not self.active_utterance():
            raise RuntimeError(
                "No active LILAC utterance. Send an initial instruction from runtime_language_cli.py."
            )
        return self.language_dataset.get(self.active_utterance())

    def decode_action(self, state, z):
        if self.model is None:
            raise RuntimeError("LILAC model is required; deterministic fallback has been removed.")
        if self.language_index is None:
            raise RuntimeError("Canonical language_index is required for precomputed SBERT embeddings.")

        entry = self.active_entry()
        alpha = float(self.alpha_labeler(entry.id))
        z_raw = np.asarray(z, dtype=np.float64).reshape(LATENT_DIM,)
        z_model = apply_latent_alignment(
            z_raw,
            self.latent_alignment,
            clip=self.clip_aligned_latent,
        )
        raw_action = self._decode_with_model(state, z_model, entry, alpha)

        raw_action = np.asarray(raw_action, dtype=np.float64).reshape(ACTION_DIM,)
        action = raw_action.copy()
        action[:3] = action[:3] * self.action_pos_scale
        action[3:] = action[3:] * self.action_rot_scale

        return action, {
            "source": "model",
            "utterance": entry.text,
            "canonical_id": entry.id,
            "alpha": alpha,
            "z": z_model.copy(),
            "z_raw": z_raw.copy(),
            "raw_action": raw_action.copy(),
            "language_stack": self.language_stack.as_dict(),
        }

    def update_target(self, T_curr, state, z):
        action, info = self.decode_action(state=state, z=z)
        T_next = apply_ee_delta_to_T(T_curr, action)
        info["action"] = action.copy()
        self.last_error = None
        return T_next, info

    def safe_update_target(self, T_curr, state, z):
        """
        Return the current target unchanged when the model path is unavailable.
        """
        try:
            return self.update_target(T_curr=T_curr, state=state, z=z)
        except Exception as exc:
            self.last_error = str(exc)
            info = self._error_info(exc, z)
            return np.asarray(T_curr, dtype=np.float64).copy(), info

    def _error_info(self, exc, z):
        return {
            "source": "error",
            "error": str(exc),
            "utterance": self.active_utterance(),
            "alpha": np.nan,
            "z": np.asarray(z, dtype=np.float64).reshape(LATENT_DIM,),
            "action": np.zeros(ACTION_DIM, dtype=np.float64),
            "language_stack": self.language_stack.as_dict(),
        }

    def _decode_with_model(self, state, z, entry, alpha):
        import torch

        query = self.language_index.lookup(entry.id)
        embedding = query["embedding"]

        device = next(self.model.parameters()).device
        state_t = torch.as_tensor(np.asarray(state, dtype=np.float32)[None, :], device=device)
        lang_t = torch.as_tensor(np.asarray(embedding, dtype=np.float32)[None, :], device=device)
        alpha_t = torch.as_tensor(np.asarray([alpha], dtype=np.float32), device=device)
        z_t = torch.as_tensor(np.asarray(z, dtype=np.float32)[None, :], device=device)

        self.model.eval()
        with torch.no_grad():
            action = self.model.decoder(state_t, lang_t, alpha_t, z_t)
        return action.detach().cpu().numpy().reshape(ACTION_DIM,)


class FileLanguageCommandSource:
    """
    Non-blocking text-file command source for online language updates.

    Supported line formats:
        utterance remote controller to box
        utterance no to the right
        instruction Pick up the cup and pour water into the bowl.
        push up
        pop
        clear

    The `utterance` command selects over all canonical utterances. Exact
    canonical text/id/alias matches bypass Gemini; only novel text calls Gemini.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.offset = 0
        self.last_poll_time = 0.0

    def poll(self, controller):
        now = time.time()
        self.last_poll_time = now

        with self.path.open("r", encoding="utf-8") as f:
            f.seek(self.offset)
            lines = f.readlines()
            self.offset = f.tell()

        events = []
        for line in lines:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            cmd, _, payload = raw.partition(" ")
            cmd = cmd.lower().strip()
            payload = payload.strip()

            try:
                if cmd in {"utterance", "say", "language"} and payload:
                    entry, event_type = controller.apply_utterance(payload)
                    events.append((event_type, entry.text))
                elif cmd in {"instruction", "set"} and payload:
                    entry = controller.set_instruction(payload, canonicalize=True)
                    events.append(("instruction", entry.text))
                elif cmd in {"push", "correction"} and payload:
                    entry = controller.push_correction(payload, canonicalize=True)
                    events.append(("push", entry.text))
                elif cmd == "pop":
                    popped = controller.pop_correction()
                    events.append(("pop", popped))
                elif cmd == "clear":
                    controller.language_stack.clear()
                    events.append(("clear", None))
                else:
                    entry, event_type = controller.apply_utterance(raw)
                    events.append((event_type, entry.text))
            except Exception as exc:
                events.append(("error", "%s: %s" % (raw, exc)))
        return events


def load_canonical_language_index(path):
    return CanonicalLanguageIndex.load(path)
