from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from std_srvs.srv import Trigger

from lilac_interfaces.msg import LatentZ

from .common import stamp


class LatentInputNode(Node):
    def __init__(self):
        super().__init__("latent_input_node")
        self.declare_parameter("publish_hz", 30.0)
        self.declare_parameter("axis_lr", 0)
        self.declare_parameter("axis_ud", 1)
        self.declare_parameter("deadzone", 0.15)
        self.declare_parameter("button_reset", 0)
        self.declare_parameter("button_pop", 1)
        self.latest = [0.0, 0.0]
        self.previous_buttons = []

        self.publisher = self.create_publisher(LatentZ, "/vader5/latent_z", 10)
        self.create_subscription(Joy, "/joy", self.on_joy, 10)
        self.reset_client = self.create_client(Trigger, "/sim/reset")
        self.pop_client = self.create_client(Trigger, "/lilac/pop_language")
        hz = float(self.get_parameter("publish_hz").value)
        self.create_timer(1.0 / max(hz, 1.0), self.publish)

    def deadzone(self, value):
        threshold = float(self.get_parameter("deadzone").value)
        return 0.0 if abs(value) < threshold else float(value)

    def on_joy(self, msg):
        lr = int(self.get_parameter("axis_lr").value)
        ud = int(self.get_parameter("axis_ud").value)
        self.latest = [
            -self.deadzone(msg.axes[lr]) if lr < len(msg.axes) else 0.0,
            -self.deadzone(msg.axes[ud]) if ud < len(msg.axes) else 0.0,
        ]
        previous = self.previous_buttons
        self.previous_buttons = list(msg.buttons)
        self.call_on_rising_edge(msg, previous, "button_reset", self.reset_client)
        self.call_on_rising_edge(msg, previous, "button_pop", self.pop_client)

    def call_on_rising_edge(self, msg, previous, parameter, client):
        index = int(self.get_parameter(parameter).value)
        current = msg.buttons[index] if index < len(msg.buttons) else 0
        before = previous[index] if index < len(previous) else 0
        if current == 1 and before == 0 and client.service_is_ready():
            client.call_async(Trigger.Request())

    def publish(self):
        msg = stamp(LatentZ(), self)
        msg.z = self.latest
        msg.raw = self.latest
        msg.source = "sensor_msgs/Joy"
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LatentInputNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
