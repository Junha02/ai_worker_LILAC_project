"""
real_zed.py

Small ZED RGB subscriber used by LILAC real-robot HRI notebooks.
"""

from __future__ import annotations

import io

import numpy as np


class ZedRGBSubscriber:
    """
    Receive JPEG RGB frames from the SH5 ZED outbound publisher.
    """

    def __init__(self, ip, port, verbose=True, conflate=True, hwm=1):
        import zmq

        self.ip = ip
        self.port = int(port)
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.SUB)

        if conflate:
            self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.set_hwm(int(hwm))
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.connect("tcp://%s:%d" % (self.ip, self.port))

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)

        if verbose:
            print("[ZED SUB] Connected to %s:%d" % (self.ip, self.port))

    def has_msg(self, timeout_ms=0):
        import zmq

        events = dict(self.poller.poll(int(timeout_ms)))
        return events.get(self.socket) == zmq.POLLIN

    def recv(self, nonblock=True, decode_image=False):
        if nonblock and not self.has_msg():
            return None
        msg = self.socket.recv_pyobj()
        return self.parse_msg(msg, decode_image=decode_image)

    def recv_latest(self, decode_image=False):
        msg_latest = None
        recv_count = 0
        while self.has_msg():
            msg_latest = self.socket.recv_pyobj()
            recv_count += 1

        if msg_latest is None:
            return None, recv_count
        return self.parse_msg(msg_latest, decode_image=decode_image), recv_count

    def parse_msg(self, msg, decode_image=False):
        if not isinstance(msg, dict):
            return None

        image_bytes = msg.get("data", None)
        image_rgb = None
        if decode_image and image_bytes is not None:
            image_rgb = decode_jpeg_rgb(image_bytes)

        return {
            "format": msg.get("format", None),
            "timestamp": msg.get("timestamp", None),
            "cam_head": {
                "name": "zed_rgb",
                "format": msg.get("format", None),
                "data": image_bytes,
                "image_rgb": image_rgb,
            },
            "raw": msg,
        }

    def get_image_rgb(self, obs):
        if obs is None:
            return None
        return obs["cam_head"]["image_rgb"]

    def close(self):
        try:
            self.poller.unregister(self.socket)
        except Exception:
            pass
        self.socket.close(linger=0)


def decode_jpeg_rgb(image_bytes):
    """
    Decode JPEG bytes to an RGB uint8 image.
    """
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as img:
        return np.asarray(img.convert("RGB")).copy()
