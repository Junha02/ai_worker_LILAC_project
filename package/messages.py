from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np


def now():
    return float(time.time())


@dataclass
class Header:
    stamp: float = field(default_factory=now)
    frame_id: str = "world"


@dataclass
class LatentZMsg:
    z: np.ndarray
    raw: np.ndarray
    source: str = "vader5"
    header: Header = field(default_factory=Header)


@dataclass
class ActiveLanguageMsg:
    text: str
    canonical_id: str
    kind: str
    alpha: float
    stack: dict
    header: Header = field(default_factory=Header)


@dataclass
class LILACStateMsg:
    q_arm: np.ndarray
    ee_pose: np.ndarray
    object_state: np.ndarray
    header: Header = field(default_factory=Header)

    @property
    def vector(self):
        return np.concatenate([self.q_arm, self.ee_pose, self.object_state]).astype(np.float32)


@dataclass
class EEDeltaMsg:
    action: np.ndarray
    raw_action: np.ndarray
    z: np.ndarray
    z_raw: np.ndarray
    language: ActiveLanguageMsg
    header: Header = field(default_factory=Header)


@dataclass
class EETargetMsg:
    T: np.ndarray
    source_action: np.ndarray
    header: Header = field(default_factory=Header)


@dataclass
class QPosCommandMsg:
    q_arm: np.ndarray
    q_hand: np.ndarray
    right_grasp: float
    ik_error: float
    header: Header = field(default_factory=Header)


@dataclass
class QPosStateMsg:
    q_arm: np.ndarray
    q_hand: np.ndarray
    header: Header = field(default_factory=Header)


@dataclass
class ContactInfoMsg:
    contact_info: dict
    robot_contact_info: dict
    header: Header = field(default_factory=Header)

    @property
    def n_all(self):
        return int(self.contact_info.get("n_contact", 0))

    @property
    def n_robot(self):
        return int(self.robot_contact_info.get("n_contact", 0))


@dataclass
class RumbleMsg:
    enabled: bool
    low: float = 0.0
    high: float = 1.0
    duration_ms: int = 120
    reason: str = ""
    header: Header = field(default_factory=Header)


@dataclass
class LILACDebugMsg:
    active_language: str
    z: np.ndarray
    action: np.ndarray
    source: str
    error: str = ""
    header: Header = field(default_factory=Header)


@dataclass
class UtteranceRequest:
    text: str
    command: str = "utterance"


@dataclass
class UtteranceResponse:
    success: bool
    event: str
    canonical_id: str = ""
    canonical_text: str = ""
    kind: str = ""
    active_text: str = ""
    message: str = ""
    stack: dict = field(default_factory=dict)
