from __future__ import annotations

from pathlib import Path
import json
import threading

import numpy as np
import torch

from paths import CANONICAL_LANGUAGE_DATASET, LILAC_RUN_DIR, RUNTIME_UTTERANCE_FILE
from messages import (
    ActiveLanguageMsg,
    ContactInfoMsg,
    EEDeltaMsg,
    EETargetMsg,
    LILACDebugMsg,
    LILACStateMsg,
    LatentZMsg,
    QPosCommandMsg,
    QPosStateMsg,
    RumbleMsg,
    UtteranceRequest,
    UtteranceResponse,
)
from ros_graph import ROSNode

from constants import ACTION_DIM, LATENT_DIM, VADER5_BUTTON_A, VADER5_BUTTON_B, VADER5_BUTTON_STT_TRIGGER
from controller import apply_ee_delta_to_T, apply_latent_alignment, load_latent_alignment
from data import T_to_pose6
from language import CanonicalLanguageDataset, CanonicalLanguageIndex, GeminiUtteranceSelector, LanguageStack
from lilac_model import LILACModel
from sh5_right_arm import (
    DEFAULT_RESET_OBJECT_BODY_NAMES,
    DEFAULT_TASK_OBJECT_SITE_NAMES,
    build_right_arm_ik_solver,
    build_vader5_handler,
    get_right_arm_joint_names,
    get_right_finger_joint_names,
    get_right_finger_qpos,
    get_right_palm_T,
    get_task_object_state,
    reset_freejoint_bodies_to_xml_pose,
    solve_right_palm_ik,
    update_right_finger_command,
    vader5_state_to_latent_z,
)


class LanguageManagerNode(ROSNode):
    def __init__(self, graph, dataset_path=CANONICAL_LANGUAGE_DATASET):
        super().__init__(graph, "language_manager_node")
        self.dataset = CanonicalLanguageDataset.load(dataset_path)
        self.selector = GeminiUtteranceSelector(self.dataset)
        self.stack = LanguageStack()
        self.pub_active = self.create_publisher("/lilac/active_language")
        self.create_service("/lilac/apply_utterance", self.apply_utterance)
        self.create_service("/lilac/pop_language", self.pop_language)
        self.create_service("/lilac/clear_language", self.clear_language)

    def apply_utterance(self, request):
        text = str(request.text).strip()
        command = str(request.command or "utterance").strip().lower()
        try:
            if command in {"instruction", "set"}:
                entry = self.selector.select(text, kind="instruction")
                self.stack.set_instruction(entry.text)
                event = "instruction"
            elif command in {"push", "correction"}:
                entry = self.selector.select(text, kind="correction")
                self.stack.push(entry.text)
                event = "push"
            else:
                entry = self.selector.select(text, kind=None)
                if entry.kind == "instruction":
                    self.stack.set_instruction(entry.text)
                    event = "instruction"
                else:
                    self.stack.push(entry.text)
                    event = "push"
            msg = self.active_message()
            self.pub_active(msg)
            return UtteranceResponse(True, event, entry.id, entry.text, entry.kind, msg.text, "ok", msg.stack)
        except Exception as exc:
            return UtteranceResponse(False, "error", message=str(exc), stack=self.stack.as_dict())

    def pop_language(self, request=None):
        popped = self.stack.pop()
        msg = self.active_message()
        self.pub_active(msg)
        return UtteranceResponse(True, "pop", active_text=msg.text, message=str(popped), stack=msg.stack)

    def clear_language(self, request=None):
        self.stack.clear()
        msg = self.active_message()
        self.pub_active(msg)
        return UtteranceResponse(True, "clear", active_text=msg.text, stack=msg.stack)

    def active_message(self):
        active = self.stack.active()
        if not active:
            return ActiveLanguageMsg("", "", "", np.nan, self.stack.as_dict())
        entry = self.dataset.get(active)
        return ActiveLanguageMsg(entry.text, entry.id, entry.kind, float(entry.alpha), self.stack.as_dict())


class UtteranceClientNode(ROSNode):
    def __init__(self, graph, request_file=RUNTIME_UTTERANCE_FILE):
        super().__init__(graph, "utterance_client_node")
        self.request_file = Path(request_file)
        self.request_file.parent.mkdir(parents=True, exist_ok=True)
        self.request_file.touch(exist_ok=True)
        self.offset = self.request_file.stat().st_size
        self.call_apply = self.create_client("/lilac/apply_utterance")
        self.call_pop = self.create_client("/lilac/pop_language")
        self.call_clear = self.create_client("/lilac/clear_language")

    def send(self, text, command="utterance"):
        if command == "pop":
            return self.call_pop(UtteranceRequest(""))
        if command == "clear":
            return self.call_clear(UtteranceRequest(""))
        return self.call_apply(UtteranceRequest(text=text, command=command))

    def poll_file(self):
        with self.request_file.open("r", encoding="utf-8") as f:
            f.seek(self.offset)
            lines = f.readlines()
            self.offset = f.tell()
        responses = []
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cmd, _, payload = line.partition(" ")
            cmd = cmd.strip().lower()
            payload = payload.strip()
            if cmd in {"utterance", "instruction", "set", "push", "correction"} and payload:
                responses.append(self.send(payload, command=cmd))
            elif cmd in {"pop", "clear"}:
                responses.append(self.send("", command=cmd))
            else:
                responses.append(self.send(line, command="utterance"))
        return responses


class JoyLatentNode(ROSNode):
    def __init__(self, graph, deadzone=0.15, joystick_idx=0, use_vader5=True):
        super().__init__(graph, "joy_latent_node")
        self.deadzone = float(deadzone)
        self.pub_latent = self.create_publisher("/vader5/latent_z")
        self.vader5 = build_vader5_handler(joystick_idx=joystick_idx, verbose=True) if use_vader5 else None
        self.js = None if self.vader5 is None else self.vader5.js
        self.prev_a = 0
        self.prev_b = 0
        self.prev_stt = 0

    def tick(self, z_override=None):
        if z_override is not None:
            z_raw = np.asarray(z_override, dtype=np.float64).reshape(LATENT_DIM,)
            msg = LatentZMsg(z=z_raw.copy(), raw=z_raw.copy(), source="manual")
            self.pub_latent(msg)
            return msg, {}

        if self.vader5 is None:
            z_raw = np.zeros(LATENT_DIM, dtype=np.float64)
            msg = LatentZMsg(z=z_raw.copy(), raw=z_raw.copy(), source="zero")
            self.pub_latent(msg)
            return msg, {}

        state = self.vader5.get_state(update_first=True)
        z_raw = vader5_state_to_latent_z(state, th=self.deadzone)
        msg = LatentZMsg(z=z_raw.copy(), raw=z_raw.copy(), source="vader5")
        self.pub_latent(msg)

        a = int(self.js.get_button(VADER5_BUTTON_A))
        b = int(self.js.get_button(VADER5_BUTTON_B))
        stt = int(self.js.get_button(VADER5_BUTTON_STT_TRIGGER))
        events = {
            "a_pressed": a == 1 and self.prev_a == 0,
            "b_pressed": b == 1 and self.prev_b == 0,
            "stt_pressed": stt == 1 and self.prev_stt == 0,
            "state": state,
        }
        self.prev_a, self.prev_b, self.prev_stt = a, b, stt
        return msg, events

    def rumble(self, msg):
        if self.vader5 is not None and msg.enabled:
            self.vader5.rumble(msg.low, msg.high, msg.duration_ms)

    def close(self):
        if self.vader5 is not None:
            self.vader5.close()


class StateAggregatorNode(ROSNode):
    def __init__(self, graph, env, object_site_names=DEFAULT_TASK_OBJECT_SITE_NAMES):
        super().__init__(graph, "state_aggregator_node")
        self.env = env
        self.object_site_names = object_site_names
        self.use_r_joint_names = get_right_arm_joint_names(env)
        self.pub_state = self.create_publisher("/lilac/state")

    def tick(self):
        q_arm = self.env.get_qpos(joint_names=self.use_r_joint_names)
        ee_pose = T_to_pose6(get_right_palm_T(self.env))
        object_state = get_task_object_state(self.env, site_names=self.object_site_names)
        msg = LILACStateMsg(q_arm=q_arm, ee_pose=ee_pose, object_state=object_state)
        self.pub_state(msg)
        return msg


class LILACPolicyNode(ROSNode):
    def __init__(
            self,
            graph,
            run_dir=LILAC_RUN_DIR,
            dataset_path=CANONICAL_LANGUAGE_DATASET,
            action_pos_scale=0.01,
            action_rot_scale=0.04,
        ):
        super().__init__(graph, "lilac_policy_node")
        self.run_dir = Path(run_dir)
        self.dataset = CanonicalLanguageDataset.load(dataset_path)
        self.model, self.model_config = LILACModel.load_bundle(self.run_dir)
        self.language_index = CanonicalLanguageIndex.load(self.run_dir / "language_index.npz")
        self.latent_alignment = None
        if (self.run_dir / "latent_alignment.npz").exists():
            self.latent_alignment = load_latent_alignment(self.run_dir / "latent_alignment.npz")
        self.action_pos_scale = float(action_pos_scale)
        self.action_rot_scale = float(action_rot_scale)
        self.latest_state = None
        self.latest_z = None
        self.latest_language = None
        self.pub_delta = self.create_publisher("/lilac/ee_delta_6d")
        self.pub_debug = self.create_publisher("/debug/lilac")
        self.create_subscription("/lilac/state", self.on_state)
        self.create_subscription("/vader5/latent_z", self.on_latent)
        self.create_subscription("/lilac/active_language", self.on_language)

    def on_state(self, msg):
        self.latest_state = msg

    def on_latent(self, msg):
        self.latest_z = msg

    def on_language(self, msg):
        self.latest_language = msg

    def tick(self):
        if self.latest_state is None or self.latest_z is None or self.latest_language is None:
            return None
        if not self.latest_language.text:
            self.pub_debug(LILACDebugMsg("", np.zeros(2), np.zeros(6), "waiting", "no active language"))
            return None

        entry = self.dataset.get(self.latest_language.canonical_id)
        embedding = self.language_index.lookup(entry.id)["embedding"]
        z_raw = np.asarray(self.latest_z.raw, dtype=np.float64).reshape(LATENT_DIM,)
        z_model = apply_latent_alignment(z_raw, self.latent_alignment, clip=True)

        device = next(self.model.parameters()).device
        state_t = torch.as_tensor(self.latest_state.vector[None, :], dtype=torch.float32, device=device)
        lang_t = torch.as_tensor(np.asarray(embedding, dtype=np.float32)[None, :], device=device)
        alpha_t = torch.as_tensor(np.asarray([entry.alpha], dtype=np.float32), device=device)
        z_t = torch.as_tensor(np.asarray(z_model, dtype=np.float32)[None, :], device=device)

        self.model.eval()
        with torch.no_grad():
            raw_action = self.model.decoder(state_t, lang_t, alpha_t, z_t)
        raw_action = raw_action.detach().cpu().numpy().reshape(ACTION_DIM,)
        action = raw_action.copy()
        action[:3] *= self.action_pos_scale
        action[3:] *= self.action_rot_scale

        msg = EEDeltaMsg(
            action=action,
            raw_action=raw_action,
            z=z_model.copy(),
            z_raw=z_raw.copy(),
            language=self.latest_language,
        )
        self.pub_delta(msg)
        self.pub_debug(LILACDebugMsg(entry.text, z_model.copy(), action.copy(), "model"))
        return msg


class EETargetNode(ROSNode):
    def __init__(self, graph, T_initial):
        super().__init__(graph, "ee_target_node")
        self.T_initial = np.asarray(T_initial, dtype=np.float64).copy()
        self.T_target = self.T_initial.copy()
        self.pub_target = self.create_publisher("/lilac/ee_target")
        self.create_subscription("/lilac/ee_delta_6d", self.on_delta)

    def reset(self):
        self.T_target = self.T_initial.copy()
        msg = EETargetMsg(self.T_target.copy(), np.zeros(6))
        self.pub_target(msg)
        return msg

    def on_delta(self, msg):
        self.T_target = apply_ee_delta_to_T(self.T_target, msg.action)
        self.pub_target(EETargetMsg(self.T_target.copy(), msg.action.copy()))


class IKCommandNode(ROSNode):
    def __init__(self, graph, env):
        super().__init__(graph, "right_arm_ik_node")
        self.env = env
        self.use_r_joint_names = get_right_arm_joint_names(env)
        self.right_hand_joint_names = get_right_finger_joint_names()
        self.ik_solver = build_right_arm_ik_solver(env, use_r_joint_names=self.use_r_joint_names)
        self.right_grasp = 0.0
        self.latest_target = None
        self.pub_qpos = self.create_publisher("/sh5/qpos_cmd")
        self.create_subscription("/lilac/ee_target", self.on_target)

    def on_target(self, msg):
        self.latest_target = msg

    def set_grasp(self, right_grasp):
        self.right_grasp = float(np.clip(right_grasp, 0.0, 1.0))

    def tick(self):
        if self.latest_target is None:
            return None
        q_arm, info = solve_right_palm_ik(
            env=self.env,
            ik_solver=self.ik_solver,
            T_rpalm_trgt=self.latest_target.T,
            use_r_joint_names=self.use_r_joint_names,
        )
        q_hand = get_right_finger_qpos(self.right_grasp)
        msg = QPosCommandMsg(q_arm=q_arm.copy(), q_hand=q_hand.copy(), right_grasp=self.right_grasp, ik_error=float(info["ik_err_best"]))
        self.pub_qpos(msg)
        return msg


class MuJoCoSimNode(ROSNode):
    def __init__(self, graph, env, sim_nstep=1):
        super().__init__(graph, "mujoco_sim_node")
        self.env = env
        self.sim_nstep = int(sim_nstep)
        self.use_r_joint_names = get_right_arm_joint_names(env)
        self.right_hand_joint_names = get_right_finger_joint_names()
        self.idxs_ctrl_arm = env.get_ctrl_idxs_attached_joints(self.use_r_joint_names)
        self.idxs_ctrl_hand = env.get_ctrl_idxs_attached_joints(self.right_hand_joint_names)
        self.ctrl_vals = env.get_ctrl_qpos(ctrl_names=env.ctrl_names)
        self.latest_qpos_cmd = None
        self.pub_contact = self.create_publisher("/sim/contact_info")
        self.pub_qpos_state = self.create_publisher("/sh5/qpos_state")
        self.create_subscription("/sh5/qpos_cmd", self.on_qpos_cmd)

    def on_qpos_cmd(self, msg):
        self.latest_qpos_cmd = msg

    def tick(self):
        if self.latest_qpos_cmd is not None:
            self.ctrl_vals[self.idxs_ctrl_arm] = self.latest_qpos_cmd.q_arm
            self.ctrl_vals[self.idxs_ctrl_hand] = self.latest_qpos_cmd.q_hand
        self.env.compensate_gravity(root_body_names=["base_link"], on=True)
        self.env.step(ctrl=self.ctrl_vals, nstep=self.sim_nstep)
        self.pub_qpos_state(QPosStateMsg(
            q_arm=self.env.get_qpos(joint_names=self.use_r_joint_names).copy(),
            q_hand=self.env.get_qpos(joint_names=self.right_hand_joint_names).copy(),
        ))
        contact_info = self.env.get_contact_info(forward=False)
        robot_contact_info = filter_robot_asset_contact_info(self.env, contact_info)
        msg = ContactInfoMsg(contact_info=contact_info, robot_contact_info=robot_contact_info)
        self.pub_contact(msg)
        return msg


class ContactHapticNode(ROSNode):
    def __init__(self, graph, env=None, low=0.0, high=1.0, duration_ms=120):
        super().__init__(graph, "contact_haptic_node")
        self.env = env
        self.low = float(low)
        self.high = float(high)
        self.duration_ms = int(duration_ms)
        self.latest_contact = None
        self.pub_rumble = self.create_publisher("/vader5/rumble")
        self.create_subscription("/sim/contact_info", self.on_contact)

    def on_contact(self, msg):
        self.latest_contact = msg
        enabled = msg.n_robot > 0
        reason = summarize_contact_pair(self.env, msg.robot_contact_info) if enabled else ""
        self.pub_rumble(RumbleMsg(enabled, self.low, self.high, self.duration_ms, reason))


class DebugRenderNode(ROSNode):
    def __init__(self, graph, env):
        super().__init__(graph, "debug_render_node")
        self.env = env
        self.debug = None
        self.contact = None
        self.rumble = RumbleMsg(False)
        self.create_subscription("/debug/lilac", self.on_debug)
        self.create_subscription("/sim/contact_info", self.on_contact)
        self.create_subscription("/vader5/rumble", self.on_rumble)

    def on_debug(self, msg):
        self.debug = msg

    def on_contact(self, msg):
        self.contact = msg

    def on_rumble(self, msg):
        self.rumble = msg

    def render(self, T_target):
        if T_target is not None:
            self.env.plot_T(T=T_target, axis_len=0.12, axis_width=0.006, label="target")
        if self.debug is not None:
            self.env.viewer_text_overlay("ROS", "topic/service LILAC sim", loc="top left")
            self.env.viewer_text_overlay("Lang", self.debug.active_language[:42], loc="top left")
            self.env.viewer_text_overlay("z", "[%+.2f %+.2f]" % (self.debug.z[0], self.debug.z[1]), loc="top left")
            self.env.viewer_text_overlay(
                "dxyz",
                "[%+.3f %+.3f %+.3f]" % (self.debug.action[0], self.debug.action[1], self.debug.action[2]),
                loc="top left",
            )
        if self.contact is not None:
            self.env.viewer_text_overlay(
                "Contact / Rumble",
                "robot:%d all:%d %s" % (self.contact.n_robot, self.contact.n_all, "ON" if self.rumble.enabled else "OFF"),
                loc="top right",
            )
            if self.contact.n_robot > 0:
                self.env.viewer_text_overlay("Robot contact", self.rumble.reason[:48], loc="top right")
                self.env.plot_contact_info(self.contact.robot_contact_info)
        self.env.render()


class STTServerNode(ROSNode):
    def __init__(self, graph, stt=None):
        super().__init__(graph, "stt_server_node")
        self.stt = stt
        self.call_apply = self.create_client("/lilac/apply_utterance")
        self.busy = False
        self.last_text = ""
        self.last_error = ""

    def transcribe_async(self, duration_sec=4.0, sample_rate=16000, input_device=None):
        if self.stt is None or self.busy:
            return False
        self.busy = True

        def worker():
            try:
                self.stt.load()
                audio = self.stt.record_audio(duration_sec=duration_sec, sample_rate=sample_rate, input_device=input_device)
                text, _ = self.stt.transcribe_audio(audio=audio, sample_rate=sample_rate)
                self.last_text = str(text).strip()
                if self.last_text:
                    self.call_apply(UtteranceRequest(self.last_text, "utterance"))
            except Exception as exc:
                self.last_error = str(exc)
            finally:
                self.busy = False

        threading.Thread(target=worker, daemon=True).start()
        return True


ROBOT_CONTACT_BODY_PREFIXES = (
    "base_link",
    "lift_link",
    "arm_base_link",
    "head_link",
    "arm_l_link",
    "arm_r_link",
    "hx5_l_base",
    "hx5_r_base",
    "finger_l_link",
    "finger_r_link",
)


def is_robot_body(body_name):
    return str(body_name).startswith(ROBOT_CONTACT_BODY_PREFIXES)


def filter_robot_asset_contact_info(env, contact_info):
    n_all = int(contact_info.get("n_contact", 0))
    keep_idxs = []
    for c_idx in range(n_all):
        body1 = env.body_names[int(contact_info["body1_idx_list"][c_idx])]
        body2 = env.body_names[int(contact_info["body2_idx_list"][c_idx])]
        if is_robot_body(body1) or is_robot_body(body2):
            keep_idxs.append(c_idx)

    filtered = {}
    for key, value in contact_info.items():
        if key == "n_contact":
            continue
        if isinstance(value, list) and len(value) == n_all:
            filtered[key] = [value[idx] for idx in keep_idxs]
        else:
            filtered[key] = value
    filtered["n_contact"] = len(keep_idxs)
    filtered["min_contact_dist"] = (
        min(float(contact_info["contact_list"][idx].dist) for idx in keep_idxs)
        if keep_idxs
        else np.inf
    )
    return filtered


def summarize_contact_pair(env, contact_info):
    if int(contact_info.get("n_contact", 0)) <= 0:
        return ""
    b1 = int(contact_info["body1_idx_list"][0])
    b2 = int(contact_info["body2_idx_list"][0])
    if env is not None:
        return "%s / %s" % (env.body_names[b1], env.body_names[b2])
    return "%d/%d" % (b1, b2)


def reset_task_objects(env, body_names=DEFAULT_RESET_OBJECT_BODY_NAMES):
    valid = [name for name in body_names if name in env.body_names]
    reset_freejoint_bodies_to_xml_pose(env, valid)
