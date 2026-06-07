from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node

from lilac_interfaces.msg import EEDelta, QPosCommand

from .common import stamp


class IKBridgeNode(Node):
    """Small ROS adapter; replace the approximation with the SH5 IK solver on the robot."""

    def __init__(self):
        super().__init__("right_arm_ik_node")
        self.q_arm = np.asarray([1.12, -0.14, -0.15, -2.48, -0.18, -0.22, -0.02], dtype=np.float32)
        self.publisher = self.create_publisher(QPosCommand, "/sh5/qpos_cmd", 10)
        self.create_subscription(EEDelta, "/lilac/ee_delta_6d", self.on_delta, 10)

    def on_delta(self, delta):
        action = np.asarray(delta.action, dtype=np.float32)
        self.q_arm[:6] += action
        msg = stamp(QPosCommand(), self)
        msg.q_arm = self.q_arm.tolist()
        msg.q_hand = [0.0] * 15
        msg.right_grasp = 0.0
        msg.ik_error = 0.0
        msg.source_action = action.tolist()
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = IKBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
