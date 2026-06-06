from __future__ import annotations

import time

import numpy as np

from paths import SCENE_XML
from messages import RumbleMsg
from nodes import (
    ContactHapticNode,
    DebugRenderNode,
    EETargetNode,
    IKCommandNode,
    JoyLatentNode,
    LanguageManagerNode,
    LILACPolicyNode,
    MuJoCoSimNode,
    StateAggregatorNode,
    UtteranceClientNode,
    reset_task_objects,
)
from ros_graph import InProcessROSGraph
from sh5_right_arm import (
    DEFAULT_TASK_OBJECT_SITE_NAMES,
    get_right_arm_joint_names,
    get_right_finger_joint_names,
    get_right_finger_qpos,
    get_right_palm_T,
    update_right_finger_command,
)


class LILACROSSimRuntime:
    def __init__(
            self,
            use_vader5=True,
            initial_utterance="Pick up the cup and pour water into the bowl.",
            inference_hz=30,
            render_hz=30,
            language_hz=5,
            vader5_deadzone=0.15,
            action_pos_scale=0.01,
            action_rot_scale=0.04,
            hand_preset_enabled=True,
            hand_grip_speed=2.0,
        ):
        self.use_vader5 = bool(use_vader5)
        self.initial_utterance = str(initial_utterance).strip() if initial_utterance else ""
        self.inference_hz = float(inference_hz)
        self.render_hz = float(render_hz)
        self.language_hz = float(language_hz)
        self.vader5_deadzone = float(vader5_deadzone)
        self.action_pos_scale = float(action_pos_scale)
        self.action_rot_scale = float(action_rot_scale)
        self.hand_preset_enabled = bool(hand_preset_enabled)
        self.hand_grip_speed = float(hand_grip_speed)

        self.env = None
        self.graph = None
        self.nodes = {}
        self.sim_nstep = 1
        self.right_grasp = 0.0
        self.right_grasp_init = 0.0
        self.T_initial = None
        self.viewer_ready = False

        self.build_scene()
        self.build_graph()

    def build_scene(self):
        from ri_motion_v5_package.mujoco_sim import MuJoCoParser
        from ri_motion_v5_package.utility import group_by_prefix

        env = MuJoCoParser(rel_xml_path=SCENE_XML, verbose=False)
        ctrl_group = group_by_prefix(
            items=env.ctrl_names,
            prefix_dict={
                "lift": "lift",
                "arm": ["arm_l", "arm_r"],
                "hand": ["finger_l", "finger_r"],
            },
        )
        ctrl_names_fd = ctrl_group["lift"] + ctrl_group["arm"] + ctrl_group["hand"]
        env.restore_ctrl_info(ctrl_names=ctrl_names_fd, verbose=False)
        env.set_ctrl_info(ctrl_names=ctrl_group["lift"], p_gain=1e6, d_gain=100.0, force=2e5, verbose=False)
        env.set_ctrl_info(ctrl_names=ctrl_group["arm"], p_gain=3000.0, d_gain=100.0, force=200.0, verbose=False)
        env.set_ctrl_info(ctrl_names=ctrl_group["hand"], p_gain=100.0, d_gain=10.0, force=None, verbose=False)
        self.sim_nstep = max(1, int(round(1.0 / (self.inference_hz * env.dt))))

        env.reset()
        env.set_p("base_link", "body", (0.0, 0.0, 0.01))
        env.forward(q=[-0.1], joint_names=["lift_joint"])
        env.forward(
            q=[1.12, -0.14, -0.15, -2.48, -0.18, -0.22, -0.02],
            joint_names=[
                "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
                "arm_r_joint5", "arm_r_joint6", "arm_r_joint7",
            ],
        )
        env.forward(
            q=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            joint_names=[
                "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
                "arm_l_joint5", "arm_l_joint6", "arm_l_joint7",
            ],
        )
        reset_task_objects(env)

        use_r_joint_names = get_right_arm_joint_names(env)
        right_hand_joint_names = get_right_finger_joint_names()
        q_arm_cmd = env.get_qpos(joint_names=use_r_joint_names).copy()
        q_hand_cmd = get_right_finger_qpos(self.right_grasp_init)
        env.forward(q=q_arm_cmd, joint_names=use_r_joint_names)
        env.forward(q=q_hand_cmd, joint_names=right_hand_joint_names)
        env.set_zero_qvel()

        self.env = env
        self.T_initial = get_right_palm_T(env).copy()
        return env

    def build_graph(self):
        graph = InProcessROSGraph()
        self.graph = graph
        self.nodes["language"] = LanguageManagerNode(graph)
        self.nodes["utterance_client"] = UtteranceClientNode(graph)
        self.nodes["joy"] = JoyLatentNode(graph, deadzone=self.vader5_deadzone, use_vader5=self.use_vader5)
        self.nodes["state"] = StateAggregatorNode(graph, self.env, object_site_names=DEFAULT_TASK_OBJECT_SITE_NAMES)
        self.nodes["policy"] = LILACPolicyNode(
            graph,
            action_pos_scale=self.action_pos_scale,
            action_rot_scale=self.action_rot_scale,
        )
        self.nodes["target"] = EETargetNode(graph, self.T_initial)
        self.nodes["ik"] = IKCommandNode(graph, self.env)
        self.nodes["sim"] = MuJoCoSimNode(graph, self.env, sim_nstep=self.sim_nstep)
        self.nodes["haptic"] = ContactHapticNode(graph, env=self.env)
        self.nodes["render"] = DebugRenderNode(graph, self.env)
        graph.subscribe("joy_latent_node", "/vader5/rumble", self.nodes["joy"].rumble)

        self.nodes["target"].reset()
        if self.initial_utterance:
            response = self.nodes["utterance_client"].send(self.initial_utterance, command="utterance")
            print("[language]", response.event, response.active_text or response.message)
        return graph

    def init_viewer(self):
        self.env.init_viewer(
            title="LILAC_ROS SH5 forward-dynamics sim",
            x_offset=0.22,
            width=1.0,
            height=1.0,
            fontscale=200,
        )
        self.env.viewer.set_cam_info(-54.03, 1.35, -34.22, np.array([0.4, -0.16, 1.12]))
        self.env.viewer.set_transparency(transparent=False)
        self.env.viewer.set_geomgroup(group_2=True, group_3=False)
        self.env.viewer.set_sitegroup(group_0=False)
        self.viewer_ready = True

    def print_graph(self):
        print("Services")
        for name in sorted(self.graph.services):
            print("  " + name)
        print("Topics")
        for name in sorted(self.graph.subscribers):
            print("  " + name)

    def step(self, z_override=None):
        responses = self.nodes["utterance_client"].poll_file()
        for response in responses:
            print("[language]", response.event, response.active_text or response.message)

        joy_msg, joy_events = self.nodes["joy"].tick(z_override=z_override)
        if joy_events.get("a_pressed", False):
            self.reset_home()
            print("[home] target, hand, and task objects reset")
        if joy_events.get("b_pressed", False):
            response = self.nodes["utterance_client"].send("", command="pop")
            print("[language] pop ->", response.active_text or "<none>")

        if self.use_vader5 and self.hand_preset_enabled:
            self.right_grasp = update_right_finger_command(
                vader5=self.nodes["joy"].vader5,
                right_grasp=self.right_grasp,
                dt=1.0 / self.inference_hz,
                grip_speed=self.hand_grip_speed,
            )

        self.nodes["ik"].set_grasp(self.right_grasp)
        self.nodes["state"].tick()
        self.nodes["policy"].tick()
        self.nodes["ik"].tick()
        self.nodes["sim"].tick()
        return joy_msg

    def render(self):
        self.nodes["render"].render(self.nodes["target"].T_target)

    def reset_home(self):
        self.right_grasp = self.right_grasp_init
        self.nodes["target"].reset()
        reset_task_objects(self.env)
        if self.use_vader5:
            self.nodes["joy"].vader5.reset_motion_memory(rot_dir="yaw")
        self.nodes["joy"].rumble(RumbleMsg(False))

    def run(self):
        if not self.viewer_ready:
            self.init_viewer()

        control_dt = 1.0 / self.inference_hz
        render_dt = 1.0 / self.render_hz
        last_control = 0.0
        last_render = 0.0
        self.env.reset_wall_time()

        try:
            while self.env.is_viewer_alive():
                now = time.time()
                self.env.increase_wall_time()
                if now - last_control >= control_dt:
                    self.step()
                    last_control = now
                if now - last_render >= render_dt:
                    self.render()
                    last_render = now
                time.sleep(0.001)
        except KeyboardInterrupt:
            print("Interrupted.")
        finally:
            self.close()

    def close(self):
        try:
            self.env.close_viewer()
        except Exception:
            pass
        self.nodes["joy"].close()
