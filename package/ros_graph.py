from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict, deque


@dataclass
class TopicState:
    name: str
    last_msg: object = None
    publish_count: int = 0


class InProcessROSGraph:
    def __init__(self):
        self.topics = {}
        self.subscribers = defaultdict(list)
        self.services = {}
        self.event_log = deque(maxlen=256)

    def publisher(self, node_name, topic):
        self.topics.setdefault(topic, TopicState(topic))

        def publish(msg):
            state = self.topics.setdefault(topic, TopicState(topic))
            state.last_msg = msg
            state.publish_count += 1
            self.event_log.append(("pub", node_name, topic))
            for callback in list(self.subscribers.get(topic, [])):
                callback(msg)

        return publish

    def subscribe(self, node_name, topic, callback):
        self.topics.setdefault(topic, TopicState(topic))
        self.subscribers[topic].append(callback)
        self.event_log.append(("sub", node_name, topic))

    def service(self, node_name, name, callback):
        self.services[name] = callback
        self.event_log.append(("srv", node_name, name))

    def client(self, node_name, name):
        def call(request):
            if name not in self.services:
                raise RuntimeError("Service is not available: %s" % name)
            self.event_log.append(("call", node_name, name))
            return self.services[name](request)

        return call

    def last(self, topic, default=None):
        state = self.topics.get(topic)
        return default if state is None else state.last_msg

    def topic_counts(self):
        return {name: state.publish_count for name, state in self.topics.items()}


class ROSNode:
    def __init__(self, graph, name):
        self.graph = graph
        self.name = name

    def create_publisher(self, topic):
        return self.graph.publisher(self.name, topic)

    def create_subscription(self, topic, callback):
        self.graph.subscribe(self.name, topic, callback)

    def create_service(self, name, callback):
        self.graph.service(self.name, name, callback)

    def create_client(self, name):
        return self.graph.client(self.name, name)
