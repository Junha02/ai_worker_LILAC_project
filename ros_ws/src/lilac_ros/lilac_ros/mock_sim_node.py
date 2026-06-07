from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from lilac_interfaces.msg import ContactInfo, LILACState, QPosCommand

from .common import stamp


class MockSimNode(Node):
    """Lightweight state/contact publisher standing in for the MuJoCo process."""

    def __init__(self):
        super().__init__("mujoco_sim_node")
        self.declare_parameter("publish_hz", 30.0)
        self.state_pub = self.create_publisher(LILACState, "/lilac/state", 10)
        self.contact_pub = self.create_publisher(ContactInfo, "/sim/contact_info", 10)
        self.create_subscription(QPosCommand, "/sh5/qpos_cmd", self.on_command, 10)
        self.create_service(Trigger, "/sim/reset", self.reset)
        self.reset_state()
        hz = float(self.get_parameter("publish_hz").value)
        self.create_timer(1.0 / max(hz, 1.0), self.publish)

    def reset_state(self):
        self.q_arm = np.asarray([1.12, -0.14, -0.15, -2.48, -0.18, -0.22, -0.02], dtype=np.float32)
        self.ee_pose = np.asarray([0.45, -0.20, 1.15, 0.0, 0.0, 0.0], dtype=np.float32)
        self.object_state = np.asarray([0.35, -0.15, 1.05, 0.45, 0.05, 1.05], dtype=np.float32)

    def reset(self, request, response):
        self.reset_state()
        response.success = True
        response.message = "simulation state reset"
        return response

    def on_command(self, msg):
        if msg.q_arm:
            self.q_arm = np.asarray(msg.q_arm, dtype=np.float32)
        self.ee_pose += np.asarray(msg.source_action, dtype=np.float32)

    def publish(self):
        state = stamp(LILACState(), self)
        state.q_arm = self.q_arm.tolist()
        state.ee_pose = self.ee_pose.tolist()
        state.object_state = self.object_state.tolist()
        self.state_pub.publish(state)

        contact = stamp(ContactInfo(), self)
        outside = bool(np.any(np.abs(self.ee_pose[:2]) > 0.8) or self.ee_pose[2] < 0.75)
        contact.n_all = 1 if outside else 0
        contact.n_robot = 1 if outside else 0
        contact.robot_contact_pairs = ["arm_r_link/table"] if outside else []
        self.contact_pub.publish(contact)


def main(args=None):
    rclpy.init(args=args)
    node = MockSimNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
