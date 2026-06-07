from __future__ import annotations

import argparse

import rclpy
from rclpy.node import Node

from lilac_interfaces.srv import ApplyUtterance


class UtteranceClient(Node):
    def __init__(self):
        super().__init__("utterance_client")
        self.client = self.create_client(ApplyUtterance, "/lilac/apply_utterance")

    def send(self, text, command):
        self.client.wait_for_service(timeout_sec=5.0)
        request = ApplyUtterance.Request()
        request.text = text
        request.command = command
        return self.client.call_async(request)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="+")
    parser.add_argument("--command", default="utterance")
    parsed, ros_args = parser.parse_known_args(args=args)
    rclpy.init(args=ros_args)
    node = UtteranceClient()
    future = node.send(" ".join(parsed.text), parsed.command)
    rclpy.spin_until_future_complete(node, future)
    response = future.result()
    print(response)
    node.destroy_node()
    rclpy.shutdown()
