"""
real_sh5_zmq.py

Small SH5 ZMQ helpers for LILAC real-robot inference notebooks.
"""

from __future__ import annotations

import time

import numpy as np

from ri_motion_v5_package.utility import get_idxs, group_by_prefix


DEFAULT_Q_MSG_FORMAT = {
    "lift":   {"prefixes": ["lift_joint"]},
    "head":   {"prefixes": ["head_joint"]},
    "arm_l":  {"prefixes": ["arm_l_joint"]},
    "hand_l": {"prefixes": ["finger_l_joint"]},
    "arm_r":  {"prefixes": ["arm_r_joint"]},
    "hand_r": {"prefixes": ["finger_r_joint"]},
}


def build_body_skeleton(
        env,
        joint_names=None,
        root_body_name="base_link",
    ):
    """
    Build a compact body skeleton from bodies actuated by selected joints.
    """
    if joint_names is None:
        joint_names = env.rev_pri_joint_names
    joint_names = list(joint_names)

    body_set = set()
    if root_body_name is not None:
        body_set.add(root_body_name)

    root_body_id = None
    if root_body_name is not None:
        root_body_id = env.body_names.index(root_body_name)

    for joint_name in joint_names:
        joint_id = env.model.joint(joint_name).id
        body_id = int(env.model.jnt_bodyid[joint_id])

        while body_id > 0:
            body_name = env.body_names[body_id]
            body_set.add(body_name)

            if root_body_id is not None and body_id == root_body_id:
                break

            body_id = int(env.model.body_parentid[body_id])

    body_names = [
        body_name for body_name in env.body_names
        if body_name in body_set
    ]

    body_edges = []
    for body_name in body_names:
        body_id = env.body_names.index(body_name)
        parent_body_id = int(env.model.body_parentid[body_id])
        parent_name = env.body_names[parent_body_id]

        if parent_name in body_set:
            body_edges.append((parent_name, body_name))

    return {
        "body_names": body_names,
        "body_edges": body_edges,
        "body_set": body_set,
        "joint_names": joint_names,
        "root_body_name": root_body_name,
    }


def plot_body_skeleton(
        env,
        body_skeleton=None,
        body_names=None,
        body_edges=None,
        r_link=0.004,
        r_joint=0.012,
        rgba_link=(1.0, 0.0, 0.0, 0.45),
        rgba_joint=(1.0, 0.0, 0.0, 0.85),
        plot_names=False,
    ):
    """
    Plot a body skeleton using MuJoCo debug drawing primitives.
    """
    if body_skeleton is not None:
        body_names = body_skeleton["body_names"]
        body_edges = body_skeleton["body_edges"]

    if body_names is None or body_edges is None:
        raise ValueError("body_skeleton or both body_names/body_edges should be provided.")

    for body_fr, body_to in body_edges:
        p_fr = env.get_p(body_fr, "body")
        p_to = env.get_p(body_to, "body")
        env.plot_cylinder_fr2to(
            p_fr=p_fr,
            p_to=p_to,
            r=r_link,
            rgba=rgba_link,
        )

    for body_name in body_names:
        p = env.get_p(body_name, "body")
        env.plot_sphere(
            p=p,
            r=r_joint,
            rgba=rgba_joint,
            label=body_name if plot_names else "",
        )


def build_screw_chain(
        env,
        joint_names=None,
        root_body_name="base_link",
        name="screw chain",
        verbose=True,
    ):
    """
    Build the RI-motion style screw chain used by the real render notebook.
    """
    import mujoco
    from ri_motion_v5_package.kinematics import ScrewChain

    body_skeleton = build_body_skeleton(
        env=env,
        joint_names=joint_names,
        root_body_name=root_body_name,
    )
    body_set = body_skeleton["body_set"]

    sc = ScrewChain(name=name, verbose=verbose)
    joint_to_screw_name = {}

    for body_name in body_skeleton["body_names"]:
        body = env.model.body(body_name)
        parent_body_id = int(body.parentid[0])
        parent_body_name = env.body_names[parent_body_id]

        if body_name == root_body_name or parent_body_name not in body_set:
            parent_name = None
        else:
            parent_name = parent_body_name

        q = env.get_p(body_name, "body")
        w = np.zeros(3)

        if body.jntnum == 1:
            joint = env.model.joint(body.jntadr[0])
            joint_name = joint.name

            if joint_name in body_skeleton["joint_names"]:
                if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                    joint_id = env.joint_names.index(joint_name)
                    axis_joint = env.model.jnt_axis[joint_id]
                    R_joint = env.get_R(joint_name, "joint")
                    w = R_joint @ axis_joint
                    joint_to_screw_name[joint_name] = body_name

        sc.add_screw(
            parent_name=parent_name,
            name=body_name,
            q=q,
            w=w,
        )

    return {
        "sc": sc,
        "joint_to_screw_name": joint_to_screw_name,
        "body_skeleton": body_skeleton,
        "body_names": body_skeleton["body_names"],
        "body_edges": body_skeleton["body_edges"],
        "joint_names": body_skeleton["joint_names"],
        "root_body_name": root_body_name,
    }


def build_q_msg_layout(env, q_msg_format_dict=None, use_joint_names=None, rest_key="_unused"):
    q_msg_format_dict = DEFAULT_Q_MSG_FORMAT if q_msg_format_dict is None else q_msg_format_dict
    sim_joint_names = list(env.rev_pri_joint_names)
    joint_names_to_group = sim_joint_names if use_joint_names is None else list(use_joint_names)
    q_msg_keys = list(q_msg_format_dict)
    prefix_dict = {
        q_msg_key: q_msg_format_dict[q_msg_key]["prefixes"]
        for q_msg_key in q_msg_keys
    }

    qmsgkeys_to_usejointnames_dict = group_by_prefix(
        items       = joint_names_to_group,
        prefix_dict = prefix_dict,
        rest_key    = rest_key,
    )
    unused_joint_names = qmsgkeys_to_usejointnames_dict.pop(rest_key, [])

    if use_joint_names is None:
        use_joint_names = [
            joint_name
            for q_msg_key in q_msg_keys
            for joint_name in qmsgkeys_to_usejointnames_dict[q_msg_key]
        ]
    else:
        use_joint_names = joint_names_to_group

    qmsgkeys_to_usejointidxs_dict = {
        q_msg_key: get_idxs(
            query_list  = use_joint_names,
            domain_list = qmsgkeys_to_usejointnames_dict[q_msg_key],
        )
        for q_msg_key in q_msg_keys
    }

    return {
        "q_msg_keys": q_msg_keys,
        "sim_joint_names": sim_joint_names,
        "use_joint_names": use_joint_names,
        "unused_joint_names": unused_joint_names,
        "qmsgkeys_to_usejointnames_dict": qmsgkeys_to_usejointnames_dict,
        "qmsgkeys_to_usejointidxs_dict": qmsgkeys_to_usejointidxs_dict,
    }


def get_qmsg_from_qpos(qpos, q_msg_layout):
    qpos = np.asarray(qpos, dtype=np.float64)
    return {
        q_msg_key: qpos[q_msg_layout["qmsgkeys_to_usejointidxs_dict"][q_msg_key]].tolist()
        for q_msg_key in q_msg_layout["q_msg_keys"]
    }


def get_qpos_from_robot_state(robot_state, q_msg_layout, env=None, q_default=None):
    if robot_state is None:
        raise ValueError("Robot state is not received.")

    if q_default is None:
        qpos = env.get_qpos(joint_names=q_msg_layout["use_joint_names"]).copy()
    else:
        qpos = np.asarray(q_default, dtype=np.float64).copy()

    joints = robot_state.get("joints", {})
    q_name_to_idx = {
        joint_name: idx
        for idx, joint_name in enumerate(q_msg_layout["use_joint_names"])
    }

    for q_msg_key in q_msg_layout["q_msg_keys"]:
        q_msg_state = joints.get(q_msg_key)
        if q_msg_state is None:
            continue

        name_to_position = dict(zip(q_msg_state["name"], q_msg_state["position"]))
        for joint_name in q_msg_layout["qmsgkeys_to_usejointnames_dict"][q_msg_key]:
            if joint_name in name_to_position:
                qpos[q_name_to_idx[joint_name]] = float(name_to_position[joint_name])

    return qpos


def get_latest_robot_state(sub, timeout_sec=1.0, sleep_sec=1e-4):
    time_start = time.time()
    robot_state_latest = None
    recv_count = 0
    while time.time() - time_start < float(timeout_sec):
        robot_state = sub.recv()
        if robot_state is not None:
            robot_state_latest = robot_state
            recv_count += 1
        time.sleep(float(sleep_sec))
    return robot_state_latest, recv_count


def get_real_q_from_robot_state(robot_state, q_msg_layout):
    qpos = []
    joint_names = []
    joints = robot_state.get("joints", {})

    for q_msg_key in q_msg_layout["q_msg_keys"]:
        q_msg_state = joints.get(q_msg_key)
        if q_msg_state is None:
            continue

        name_to_position = dict(zip(q_msg_state["name"], q_msg_state["position"]))
        for joint_name in q_msg_layout["qmsgkeys_to_usejointnames_dict"][q_msg_key]:
            if joint_name in name_to_position:
                joint_names.append(joint_name)
                qpos.append(float(name_to_position[joint_name]))

    return joint_names, np.asarray(qpos, dtype=np.float64)


def send_render_qpos(
        pub_render,
        qpos,
        joint_names,
        q_msg_layout,
        msg_type,
        on_flag,
        robot_hz,
        render_hz,
        robot_pub_hz_actual,
        extra=None,
    ):
    obj = {
        "msg_type": msg_type,
        "time": time.time(),
        "joint_names": list(joint_names),
        "qpos": np.asarray(qpos, dtype=np.float64).copy(),
        "q_names_dict": q_msg_layout["qmsgkeys_to_usejointnames_dict"],
        "pub_robot": bool(on_flag),
        "robot_hz": float(robot_hz),
        "render_hz": float(render_hz),
        "robot_pub_hz_actual": float(robot_pub_hz_actual),
    }
    if extra is not None:
        obj.update(extra)
    pub_render.send(obj=obj)
    return obj
