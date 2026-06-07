from __future__ import annotations

import rclpy
from rclpy.node import Node

from lilac_interfaces.msg import ContactInfo, Rumble

from .common import stamp


class HapticNode(Node):
    def __init__(self):
        super().__init__("contact_haptic_node")
        self.publisher = self.create_publisher(Rumble, "/vader5/rumble", 10)
        self.create_subscription(ContactInfo, "/sim/contact_info", self.on_contact, 10)

    def on_contact(self, contact):
        msg = stamp(Rumble(), self)
        msg.enabled = contact.n_robot > 0
        msg.low = 0.0
        msg.high = 1.0
        msg.duration_ms = 120
        msg.reason = contact.robot_contact_pairs[0] if contact.robot_contact_pairs else ""
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HapticNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
