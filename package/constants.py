"""
constants.py

Project-wide constants for the SH5 right-arm LILAC implementation.
"""

import numpy as np


# LILAC paper constants.
CONTROL_HZ = 10
ACTION_SPACE = "ee-euler-delta"
ACTION_DIM = 6
LATENT_DIM = 2
HIDDEN_DIM = 128
LANGUAGE_DIM = 768


# SH5 right arm, fixed to 7 DoF as requested.
RIGHT_ARM_JOINT_NAMES = [
    "arm_r_joint1",
    "arm_r_joint2",
    "arm_r_joint3",
    "arm_r_joint4",
    "arm_r_joint5",
    "arm_r_joint6",
    "arm_r_joint7",
]

RIGHT_PALM_SITE_NAME = "rpalm"
RIGHT_PALM_SITE_NAMES = [
    "rpalm",
    "rpalm_top",
    "rpalm_palmar",
    "rpalm_front",
]

RIGHT_FINGER_JOINT_NAMES = [
    "finger_r_joint2",
    "finger_r_joint3",
    "finger_r_joint4",
    "finger_r_joint6",
    "finger_r_joint7",
    "finger_r_joint8",
    "finger_r_joint10",
    "finger_r_joint11",
    "finger_r_joint12",
    "finger_r_joint14",
    "finger_r_joint15",
    "finger_r_joint16",
    "finger_r_joint18",
    "finger_r_joint19",
    "finger_r_joint20",
]


# Vader5 mapping used by the SH5 right-arm collection notebooks.
VADER5_AXIS_MAP = {
    "stick_left_ud": 3,
    "stick_left_lr": 2,
    "stick_right_ud": 1,
    "trigger_top_left": 4,
    "trigger_top_right": 5,
}

VADER5_BUTTON_MAP = {
    "button_face_left": 12,
    "button_face_down": 13,
    "button_face_right": 14,
    "button_dpad_up": 11,
    "button_a": 0,
    "button_b": 1,
    "button_x": 2,
    "button_y": 3,
}

VADER5_BUTTON_A = 0
VADER5_BUTTON_B = 1
VADER5_BUTTON_X = 2
VADER5_BUTTON_Y = 3
VADER5_BUTTON_LANGUAGE_POP = 11
VADER5_BUTTON_STT_TRIGGER = 11
VADER5_BUTTON_UTTERANCE_TRIGGER = 11


# Paper action ordering: Cartesian position delta, then Euler orientation delta.
ACTION_INDEX = {
    "x": 0,
    "y": 1,
    "z": 2,
    "roll": 3,
    "pitch": 4,
    "yaw": 5,
}


# A neutral SH5 right-arm posture used only as a safe reference for nullspace targets.
RIGHT_ARM_Q_HOME = (
    np.array([0.0, 0.0, 0.0, -90.0, 0.0, 0.0, 0.0], dtype=np.float64)
    * np.pi
    / 180.0
)
