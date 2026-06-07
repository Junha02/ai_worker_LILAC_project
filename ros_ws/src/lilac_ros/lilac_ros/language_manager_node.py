from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger

from lilac_interfaces.msg import ActiveLanguage
from lilac_interfaces.srv import ApplyUtterance

from .common import add_core_package_to_path, canonical_dataset_path, json_text, stamp


add_core_package_to_path()
from language import CanonicalLanguageDataset, GeminiUtteranceSelector, LanguageStack  # noqa: E402


class LanguageManagerNode(Node):
    def __init__(self):
        super().__init__("language_manager_node")
        self.declare_parameter("dataset_path", str(canonical_dataset_path()))
        self.declare_parameter("initial_instruction", "Pick up the cup and pour water into the bowl.")

        dataset_path = self.get_parameter("dataset_path").value
        self.dataset = CanonicalLanguageDataset.load(dataset_path)
        self.selector = GeminiUtteranceSelector(self.dataset)
        self.stack = LanguageStack()

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.active_pub = self.create_publisher(ActiveLanguage, "/lilac/active_language", qos)
        self.create_service(ApplyUtterance, "/lilac/apply_utterance", self.apply_utterance)
        self.create_service(Trigger, "/lilac/pop_language", self.pop_language)
        self.create_service(Trigger, "/lilac/clear_language", self.clear_language)

        initial = str(self.get_parameter("initial_instruction").value).strip()
        if initial:
            entry = self.dataset.get(initial, kind="instruction")
            self.stack.set_instruction(entry.text)
        self.publish_active()

    def active_message(self):
        msg = stamp(ActiveLanguage(), self)
        msg.stack_json = json_text(self.stack.as_dict())
        active = self.stack.active()
        if not active:
            msg.alpha = float("nan")
            return msg
        entry = self.dataset.get(active)
        msg.text = entry.text
        msg.canonical_id = entry.id
        msg.kind = entry.kind
        msg.alpha = float(entry.alpha)
        return msg

    def publish_active(self):
        msg = self.active_message()
        self.active_pub.publish(msg)
        return msg

    def apply_utterance(self, request, response):
        command = str(request.command or "utterance").strip().lower()
        text = str(request.text).strip()
        try:
            kind = "instruction" if command in {"instruction", "set"} else None
            kind = "correction" if command in {"push", "correction"} else kind
            entry = self.selector.select(text, kind=kind)
            if entry.kind == "instruction":
                self.stack.set_instruction(entry.text)
                event = "instruction"
            else:
                self.stack.push(entry.text)
                event = "push"
            active = self.publish_active()
            response.success = True
            response.event = event
            response.canonical_id = entry.id
            response.canonical_text = entry.text
            response.kind = entry.kind
            response.active_text = active.text
            response.message = "ok"
        except Exception as exc:
            response.success = False
            response.event = "error"
            response.active_text = self.stack.active()
            response.message = str(exc)
            self.get_logger().error(str(exc))
        response.stack_json = json_text(self.stack.as_dict())
        return response

    def pop_language(self, request, response):
        popped = self.stack.pop()
        active = self.publish_active()
        response.success = True
        response.message = "popped=%s active=%s" % (popped, active.text)
        return response

    def clear_language(self, request, response):
        self.stack.clear()
        active = self.publish_active()
        response.success = True
        response.message = "active=%s" % active.text
        return response


def main(args=None):
    rclpy.init(args=args)
    node = LanguageManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
