"""
sh5_right_arm.py

ROBOTIS FFW SH5 right-arm utilities for LILAC and Vader5 control.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from constants import (
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_FINGER_JOINT_NAMES,
    RIGHT_PALM_SITE_NAME,
    VADER5_AXIS_MAP,
    VADER5_BUTTON_MAP,
)


THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parent
PROJECT_DIR = PACKAGE_DIR.parent
LAB_DIR = PROJECT_DIR.parents[1]
DEFAULT_TASK_OBJECT_SITE_NAMES = [
    "task_cup_site",
    "task_bowl_site",
]
DEFAULT_RESET_OBJECT_BODY_NAMES = [
    "task_cup",
    "task_remote",
]

REAL_INFERENCE_START_RIGHT_ARM_Q = np.array(
    [1.12, -0.14, -0.15, -2.48, -0.18, -0.22, -0.02],
    dtype=np.float64,
)
REAL_INFERENCE_START_LEFT_ARM_Q = np.zeros(7, dtype=np.float64)


def quat_wxyz_to_R(quat):
    """
    Convert a MuJoCo wxyz quaternion to a rotation matrix.
    """
    q = np.asarray(quat, dtype=np.float64).reshape(4,)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def get_xml_body_home_pR(env, body_name):
    """
    Return a body's XML-authored local pose.
    """
    body = env.model.body(body_name)
    p_home = np.asarray(body.pos, dtype=np.float64).copy()
    R_home = quat_wxyz_to_R(np.asarray(body.quat, dtype=np.float64))
    return p_home, R_home


def reset_freejoint_bodies_to_xml_pose(env, body_names):
    """
    Reset freejoint task bodies to their XML pose and clear velocity.
    """
    for body_name in body_names:
        if body_name not in getattr(env, "body_names", []):
            continue
        jntadr = int(env.model.body(body_name).jntadr[0])
        if jntadr < 0:
            continue
        p_home, R_home = get_xml_body_home_pR(env, body_name)
        env.set_pR(body_name, "base", p_home, R_home, forward=False)
    env.forward()
    env.set_zero_qvel()


def create_lilac_scene_xml(
        output_xml_path = None,
        robot_xml_path  = None,
        floor_xml_path  = None,
        extra_xml_paths = None,
        verbose         = True,
    ):
    """
    Create the SH5 MuJoCo scene used by LILAC notebooks.
    """
    from ri_motion_v5_package.mujoco_sim import merge_mjcfs

    if output_xml_path is None:
        output_xml_path = PROJECT_DIR / "notebook" / "xml" / "scene_ffw_sh5_lilac.xml"
    if floor_xml_path is None:
        floor_xml_path = LAB_DIR / "ri_motion_v5_asset" / "floor" / "floor_simple_white.xml"
    if robot_xml_path is None:
        robot_xml_path = (
            LAB_DIR
            / "ri_motion_v5_asset"
            / "robotis"
            / "robotis_ffw"
            / "ffw_sh5_site_added_fixed.xml"
        )
    if extra_xml_paths is None:
        extra_xml_paths = []

    included_mjcf_files = [
        str(floor_xml_path),
        str(robot_xml_path),
    ] + [str(path) for path in extra_xml_paths]

    return merge_mjcfs(
        included_mjcf_files = included_mjcf_files,
        output_xml_path     = str(output_xml_path),
        verbose             = verbose,
    )


def get_right_arm_joint_names(env):
    """
    Resolve right-arm joints from the SH5 MuJoCo model and verify 7 DoF.
    """
    from ri_motion_v5_package.utility import group_by_prefix

    groups = group_by_prefix(
        items       = env.rev_pri_joint_names,
        prefix_dict = {"arm_r_joint_names": ["arm_r_joint"]},
    )
    joint_names = groups["arm_r_joint_names"]
    if len(joint_names) != 7:
        raise ValueError("SH5 right arm should have 7 joints, got %d: %s" % (len(joint_names), joint_names))
    return joint_names


def get_right_finger_joint_names():
    return list(RIGHT_FINGER_JOINT_NAMES)


def get_task_object_state(env, site_names=None):
    """
    Read task object state from scene sites as flattened xyz values.
    """
    if site_names is None:
        site_names = DEFAULT_TASK_OBJECT_SITE_NAMES

    p_sites = []
    for site_name in site_names:
        if site_name not in getattr(env, "site_names", []):
            raise ValueError("Task object site is missing from the scene: %s" % site_name)
        p_sites.append(env.get_p(site_name, "site"))
    return np.concatenate(p_sites).astype(np.float64)


def update_right_finger_command(vader5, right_grasp, dt, grip_speed=2.5):
    """
    Same preset-grasp update convention as the reference Vader5 notebook.
    """
    js = vader5.js
    if js.get_button(3):
        right_grasp += grip_speed * dt
    if js.get_button(2):
        right_grasp -= grip_speed * dt
    return float(np.clip(right_grasp, 0.0, 1.0))


def get_right_finger_qpos(val_grasp):
    """
    Right-hand preset with faster thumb curl and softer finger curl.
    """
    from ri_motion_v5_package.kinematics.transforms import D2R

    grasp = float(np.clip(val_grasp, 0.0, 1.0))
    thumb_grasp = float(np.clip(grasp * 1.2, 0.0, 1.0))
    finger_grasp = grasp * 0.6
    thumb_base = np.array([-100.0])
    thumb_curl = np.array([45.0, 45.0]) * thumb_grasp
    finger_curl = np.array([90.0, 90.0, 60.0] * 4) * finger_grasp

    return np.concatenate([
        thumb_base,
        thumb_curl,
        finger_curl,
    ]) * D2R


def build_real_inference_start_qpos(q_msg_layout, right_grasp=0.0):
    """
    Full SH5 qpos target used as the final homing pose before real inference.
    """
    use_joint_names = list(q_msg_layout["use_joint_names"])
    group_idxs = q_msg_layout["qmsgkeys_to_usejointidxs_dict"]
    q = np.zeros(len(use_joint_names), dtype=np.float64)

    def set_group(q_msg_key, values):
        if q_msg_key not in group_idxs:
            return
        idxs = np.asarray(group_idxs[q_msg_key], dtype=int)
        vals = np.asarray(values, dtype=np.float64)
        if len(vals) != len(idxs):
            raise ValueError("%s target length mismatch: %d != %d" % (q_msg_key, len(vals), len(idxs)))
        q[idxs] = vals

    set_group("lift", np.zeros(len(group_idxs.get("lift", [])), dtype=np.float64))
    set_group("head", np.zeros(len(group_idxs.get("head", [])), dtype=np.float64))
    set_group("arm_l", REAL_INFERENCE_START_LEFT_ARM_Q)
    set_group("arm_r", REAL_INFERENCE_START_RIGHT_ARM_Q)
    set_group("hand_l", np.zeros(len(group_idxs.get("hand_l", [])), dtype=np.float64))
    set_group("hand_r", np.zeros(len(group_idxs.get("hand_r", [])), dtype=np.float64))

    name_to_idx = {joint_name: idx for idx, joint_name in enumerate(use_joint_names)}
    q_finger_r = get_right_finger_qpos(right_grasp)
    for joint_name, q_joint in zip(RIGHT_FINGER_JOINT_NAMES, q_finger_r):
        if joint_name in name_to_idx:
            q[name_to_idx[joint_name]] = q_joint

    return q


def build_vader5_handler(joystick_idx=0, verbose=True):
    """
    Build a Vader5 handler with the project axis/button maps.
    """
    from ri_motion_v5_package.vader5 import Vader5Handler

    return Vader5Handler(
        joystick_idx = joystick_idx,
        axis_map     = dict(VADER5_AXIS_MAP),
        button_map   = dict(VADER5_BUTTON_MAP),
        verbose      = verbose,
    )


def vader5_state_to_latent_z(state, th=0.05):
    """
    Convert the left stick to the 2-DoF latent action input z.
    """
    from ri_motion_v5_package.utility import deadzone

    z_lr = -deadzone(state.get("stick_left_lr", 0.0), th=th, center=0.0, rescale=True)
    z_ud = -deadzone(state.get("stick_left_ud", 0.0), th=th, center=0.0, rescale=True)
    return np.asarray([z_lr, z_ud], dtype=np.float64)


def build_right_arm_ik_solver(env, use_r_joint_names=None):
    """
    Build the same site-position IK solver configuration used in the reference notebook.
    """
    from ri_motion_v5_package.kinematics.transforms import D2R
    from ri_motion_v5_package.mujoco_sim import MuJoCoParser, SitePositionIKSolverRevPriBase

    if use_r_joint_names is None:
        use_r_joint_names = get_right_arm_joint_names(env)

    ik_env = MuJoCoParser(rel_xml_path=env.rel_xml_path)
    dls_damping = 1e-6 * np.ones(len(use_r_joint_names))
    dls_damping[-1] = 1e-3

    ik_solver = SitePositionIKSolverRevPriBase(
        ik_env            = ik_env,
        max_ik_tick       = 10,
        ik_stepsize_rev   = 10 * D2R,
        ik_stepsize_pri   = 0.05,
        ik_update_th_rev  = 10 * D2R,
        ik_update_th_pri  = 0.05,
        dls_damping       = dls_damping,
        max_probe_rev     = 3 * D2R,
        max_probe_pri     = 0.01,
        k_null            = 0.5,
        q_home_rev_pri    = env.get_qpos(joint_names=env.rev_pri_joint_names),
    )
    return ik_solver


def get_right_palm_T(env):
    from ri_motion_v5_package.mujoco_sim import get_T_hand_from_sites

    return get_T_hand_from_sites(
        env              = env,
        site_name_common = RIGHT_PALM_SITE_NAME,
        hand_type        = "right",
        site_type        = "site",
    )


def add_right_palm_ik_targets(ik_solver, T_rpalm_trgt):
    """
    Add palm, top, palmar, and front site targets to the IK solver.
    """
    from ri_motion_v5_package.kinematics.transforms import t2pr

    p_rpalm_trgt, R_rpalm_trgt = t2pr(T_rpalm_trgt)
    p_rpalm_top_trgt = p_rpalm_trgt + 0.1 * R_rpalm_trgt[:, 2]
    p_rpalm_palmar_trgt = p_rpalm_trgt + 0.1 * R_rpalm_trgt[:, 1]
    p_rpalm_front_trgt = p_rpalm_trgt + 0.1 * R_rpalm_trgt[:, 0]

    ik_solver.reset_buffers()
    ik_solver.add_ik_target("rpalm", p_rpalm_trgt)
    ik_solver.add_ik_target("rpalm_top", p_rpalm_top_trgt)
    ik_solver.add_ik_target("rpalm_palmar", p_rpalm_palmar_trgt)
    ik_solver.add_ik_target("rpalm_front", p_rpalm_front_trgt)


def solve_right_palm_ik(env, ik_solver, T_rpalm_trgt, use_r_joint_names=None):
    """
    Solve right-palm site IK and return (q_right_arm, info).
    """
    if use_r_joint_names is None:
        use_r_joint_names = get_right_arm_joint_names(env)

    add_right_palm_ik_targets(ik_solver, T_rpalm_trgt)
    _, info = ik_solver.solve_ik(
        env                     = env,
        joints_use              = use_r_joint_names,
        joint_limit_handle_flag = True,
        nullspace_control_flag  = True,
        base_control_flag       = False,
    )
    qpos_used_best = info["qpos_used_best"]
    if list(info["joints_use"]) != list(use_r_joint_names):
        raise RuntimeError("IK returned an unexpected joint set.")
    if len(qpos_used_best) != len(use_r_joint_names):
        raise RuntimeError("IK returned an unexpected qpos length.")
    return qpos_used_best, info


def forward_right_arm_and_hand(env, q_arm, right_grasp, use_r_joint_names=None):
    """
    Forward SH5 state for right arm plus right hand preset.
    """
    if use_r_joint_names is None:
        use_r_joint_names = list(RIGHT_ARM_JOINT_NAMES)
    q_finger_r = get_right_finger_qpos(right_grasp)
    joint_names_all = list(use_r_joint_names) + get_right_finger_joint_names()
    qpos_all = np.concatenate([np.asarray(q_arm, dtype=np.float64), q_finger_r])
    env.forward(q=qpos_all, joint_names=joint_names_all)
    return qpos_all, joint_names_all


def step_right_arm_and_hand(env, q_arm_cmd, right_grasp_cmd, use_r_joint_names=None, nstep=1):
    """
    Apply right-arm/right-hand position actuator targets and advance MuJoCo dynamics.
    """
    if use_r_joint_names is None:
        use_r_joint_names = list(RIGHT_ARM_JOINT_NAMES)
    q_finger_cmd = get_right_finger_qpos(right_grasp_cmd)
    joint_names_all = list(use_r_joint_names) + get_right_finger_joint_names()
    qpos_cmd_all = np.concatenate([np.asarray(q_arm_cmd, dtype=np.float64), q_finger_cmd])
    ctrl_qpos_names = set(getattr(env, "ctrl_qpos_names", []))
    if ctrl_qpos_names:
        mask = [name in ctrl_qpos_names for name in joint_names_all]
        step_joint_names = [name for name, keep in zip(joint_names_all, mask) if keep]
        step_qpos_cmd = qpos_cmd_all[np.asarray(mask, dtype=bool)]
    else:
        step_joint_names = joint_names_all
        step_qpos_cmd = qpos_cmd_all

    env.step(
        ctrl        = step_qpos_cmd,
        joint_names = step_joint_names,
        nstep       = int(nstep),
    )

    q_arm_actual = env.get_qpos(joint_names=use_r_joint_names)
    q_hand_actual = env.get_qpos(joint_names=get_right_finger_joint_names())
    qpos_actual_all = np.concatenate([q_arm_actual, q_hand_actual])
    return qpos_cmd_all, qpos_actual_all, joint_names_all
