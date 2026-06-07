from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from lilac_interfaces.msg import ActiveLanguage, EEDelta, LILACState, LatentZ

from .common import add_core_package_to_path, canonical_dataset_path, run_dir, stamp


add_core_package_to_path()
from controller import apply_latent_alignment, load_latent_alignment  # noqa: E402
from language import CanonicalLanguageDataset, CanonicalLanguageIndex  # noqa: E402
from lilac_model import LILACModel  # noqa: E402


class PolicyNode(Node):
    def __init__(self):
        super().__init__("lilac_policy_node")
        self.declare_parameter("run_dir", str(run_dir()))
        self.declare_parameter("dataset_path", str(canonical_dataset_path()))
        self.declare_parameter("publish_hz", 30.0)
        self.declare_parameter("action_pos_scale", 0.01)
        self.declare_parameter("action_rot_scale", 0.04)
        self.declare_parameter("demo_fallback_enabled", True)

        self.dataset = CanonicalLanguageDataset.load(self.get_parameter("dataset_path").value)
        self.latest_state = None
        self.latest_z = None
        self.latest_language = None
        self.model = None
        self.language_index = None
        self.latent_alignment = None
        self.load_model()

        self.publisher = self.create_publisher(EEDelta, "/lilac/ee_delta_6d", 10)
        self.create_subscription(LILACState, "/lilac/state", self.on_state, 10)
        self.create_subscription(LatentZ, "/vader5/latent_z", self.on_latent, 10)
        language_qos = QoSProfile(depth=1)
        language_qos.reliability = ReliabilityPolicy.RELIABLE
        language_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(
            ActiveLanguage, "/lilac/active_language", self.on_language, language_qos
        )
        hz = float(self.get_parameter("publish_hz").value)
        self.create_timer(1.0 / max(hz, 1.0), self.tick)

    def load_model(self):
        path = self.get_parameter("run_dir").value
        try:
            self.model, _ = LILACModel.load_bundle(path)
            self.language_index = CanonicalLanguageIndex.load(path + "/language_index.npz")
            alignment_path = path + "/latent_alignment.npz"
            from pathlib import Path
            if Path(alignment_path).exists():
                self.latent_alignment = load_latent_alignment(alignment_path)
            self.get_logger().info("Loaded trained LILAC model from %s" % path)
        except Exception as exc:
            self.get_logger().warning("Using ROS demo policy: %s" % exc)

    def on_state(self, msg):
        self.latest_state = msg

    def on_latent(self, msg):
        self.latest_z = msg

    def on_language(self, msg):
        self.latest_language = msg

    def tick(self):
        if self.latest_state is None or self.latest_z is None or self.latest_language is None:
            return
        if not self.latest_language.canonical_id:
            return

        z_raw = np.asarray(self.latest_z.raw, dtype=np.float32)
        try:
            if self.model is not None:
                raw_action = self.decode_model(z_raw)
                source = "lilac_model"
            elif bool(self.get_parameter("demo_fallback_enabled").value):
                raw_action = self.decode_demo(z_raw, self.latest_language.canonical_id)
                source = "demo_basis"
            else:
                return
            error = ""
        except Exception as exc:
            raw_action = np.zeros(6, dtype=np.float32)
            source = "hold"
            error = str(exc)

        msg = stamp(EEDelta(), self)
        msg.raw_action = [float(value) for value in raw_action]
        msg.action = [float(value) for value in raw_action]
        msg.z = [float(value) for value in self.latest_z.z]
        msg.z_raw = [float(value) for value in self.latest_z.raw]
        msg.canonical_id = self.latest_language.canonical_id
        msg.active_text = self.latest_language.text
        msg.alpha = self.latest_language.alpha
        msg.source = source
        msg.error = error
        self.publisher.publish(msg)

    def decode_model(self, z_raw):
        import torch

        state = np.concatenate([
            np.asarray(self.latest_state.q_arm, dtype=np.float32),
            np.asarray(self.latest_state.ee_pose, dtype=np.float32),
            np.asarray(self.latest_state.object_state, dtype=np.float32),
        ])
        entry = self.dataset.get(self.latest_language.canonical_id)
        embedding = self.language_index.lookup(entry.id)["embedding"]
        z = apply_latent_alignment(z_raw, self.latent_alignment, clip=True)
        with torch.no_grad():
            action = self.model.decoder(
                torch.as_tensor(state[None, :], dtype=torch.float32),
                torch.as_tensor(embedding[None, :], dtype=torch.float32),
                torch.as_tensor([entry.alpha], dtype=torch.float32),
                torch.as_tensor(z[None, :], dtype=torch.float32),
            ).cpu().numpy().reshape(6)
        action[:3] *= float(self.get_parameter("action_pos_scale").value)
        action[3:] *= float(self.get_parameter("action_rot_scale").value)
        return action.astype(np.float32)

    def decode_demo(self, z, canonical_id):
        x, y = float(z[0]), float(z[1])
        action = np.zeros(6, dtype=np.float32)
        bases = {
            "right": (0, -1.0), "left": (0, 1.0),
            "up": (2, 1.0), "down": (2, -1.0),
            "front": (1, 1.0), "back": (1, -1.0),
            "pour_water": (3, 1.0),
        }
        if canonical_id in bases:
            index, sign = bases[canonical_id]
            scale = 0.04 if index >= 3 else 0.01
            action[index] = sign * y * scale
        else:
            action[0] = x * 0.01
            action[1] = y * 0.01
        return action


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
