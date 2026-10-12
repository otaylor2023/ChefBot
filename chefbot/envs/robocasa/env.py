"""RoboCasaEnv: the Env interface on a RoboCasa (robosuite + MuJoCo) kitchen.

Low-level only: it accepts base velocity commands, steps physics, and reports
state. Planning and control live in chefbot/robot.
"""

from __future__ import annotations

import math
import threading

import mujoco
import numpy as np
import robocasa  # noqa: F401  (registers the kitchen environments with robosuite)
import robosuite
import robosuite.macros as robosuite_macros
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)

from chefbot.envs.robocasa.config import RoboCasaConfig
from chefbot.interface.env import Env
from chefbot.interface.types import RGBD, EnvState, Pose2D, wrap_angle

_ROBOT_PREFIXES = ("robot0_", "mobilebase0_", "gripper0_")
_BASE_BODY = "mobilebase0_base"
_BASE_JOINTS = ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side", "mobilebase0_joint_mobile_yaw")
_BASE_ACTUATORS = ("mobilebase0_actuator_mobile_forward", "mobilebase0_actuator_mobile_side", "mobilebase0_actuator_mobile_yaw")
# Below this commanded speed the friction feedforward fades out, so a zero
# command means zero actuator effort.
_FRICTION_FADE = np.array([0.005, 0.005, 0.01])


class RoboCasaEnv(Env):
    """A RoboCasa kitchen with the default mobile Franka (PandaOmron).

    The base is holonomic: three velocity actuators on x / y slide joints and a
    yaw hinge. The base pose is the yaw joint's anchor, which is also the center
    of the base footprint.
    """

    def __init__(self, config: RoboCasaConfig | None = None, render: bool = False):
        self.config = config or RoboCasaConfig.load()
        cfg = self.config
        robosuite_macros.SIMULATION_TIMESTEP = cfg.physics_timestep
        self._render = render
        # Every sim access goes through this lock: the motion thread steps
        # physics while other threads read poses and run detection.
        self.lock = threading.RLock()
        self.suite = robosuite.make(
            env_name=cfg.task,
            robots=cfg.robot,
            controller_configs=load_composite_controller_config(robot=cfg.robot),
            layout_ids=cfg.layout,
            style_ids=cfg.style,
            seed=cfg.seed,
            control_freq=cfg.control_freq,
            has_renderer=False,
            has_offscreen_renderer=render,
            use_camera_obs=False,
            ignore_done=True,
        )
        self.suite.reset()
        self.model: mujoco.MjModel = self.suite.sim.model._model
        self.data: mujoco.MjData = self.suite.sim.data._data
        model = self.model

        self._base_body = model.body(_BASE_BODY).id
        joints = [model.joint(name).id for name in _BASE_JOINTS]
        self._slide_joints, self._yaw_joint = joints[:2], joints[2]
        self._base_qpos = [int(model.jnt_qposadr[j]) for j in joints]
        self._base_dofs = [int(model.jnt_dofadr[j]) for j in joints]
        actuators = [model.actuator(name).id for name in _BASE_ACTUATORS]
        # A velocity actuator settles where kv * (ctrl - v) balances joint
        # friction, i.e. `frictionloss / kv` short of the commanded velocity.
        self._friction_offset = np.array(
            [model.dof_frictionloss[d] / model.actuator_gainprm[a, 0] for d, a in zip(self._base_dofs, actuators)]
        )
        self._ctrl_max = np.array([model.actuator_ctrlrange[a, 1] for a in actuators])

        self._controller = self.suite.robots[0].composite_controller
        self._base_controller = self._controller.part_controllers["base"]
        self._command_world = np.zeros(3)
        self.contact_ticks = 0  # control ticks that ended with the robot touching something

        self.robot_geoms = frozenset(
            g for g in range(model.ngeom) if self.is_robot_body(int(model.geom_bodyid[g]))
        )
        self._floor_geoms = frozenset(
            g for g in range(model.ngeom) if self.root_body_name(int(model.geom_bodyid[g])).startswith("floor")
        )

    # -- model helpers (shared with the detector and the map builder) ---------

    def root_body_name(self, body: int) -> str:
        """Name of the top-level body (direct child of the world) that `body` hangs from."""
        parents = self.model.body_parentid
        while parents[body] != 0:
            body = int(parents[body])
        return self.model.body(body).name

    def is_robot_body(self, body: int) -> bool:
        return self.root_body_name(body).startswith(_ROBOT_PREFIXES)

    # -- Env ------------------------------------------------------------------

    def get_state(self) -> EnvState:
        with self.lock:
            pose = self._base_pose()
            # World-frame velocity of the footprint center, from the base joints.
            qvel = self.data.qvel
            linear = sum(self.data.xaxis[j][:2] * qvel[d] for j, d in zip(self._slide_joints, self._base_dofs))
            base_velocity = np.array([linear[0], linear[1], qvel[self._base_dofs[2]]])
            return EnvState(pose, base_velocity, self._base_in_contact(), float(self.data.time))

    def set_base_velocity(self, vx: float, vy: float, vyaw: float, frame: str = "world") -> None:
        with self.lock:
            if frame == "base":
                yaw = self._base_pose().yaw
                c, s = math.cos(yaw), math.sin(yaw)
                vx, vy = c * vx - s * vy, s * vx + c * vy
            elif frame != "world":
                raise ValueError(f"unknown frame {frame!r}")
            self._command_world = np.array([vx, vy, vyaw], dtype=float)

    def set_arm_joint_targets(self, q: np.ndarray) -> None:
        raise NotImplementedError(
            "arm joint targets are not wired up in the RoboCasa backend yet: "
            "PandaOmron's default arm controller is OSC_POSE (unused by nav v0.1)"
        )

    def get_camera_rgbd(self, cam: str) -> RGBD:
        if not self._render:
            raise RuntimeError("RoboCasaEnv was created with render=False; pass render=True for cameras")
        cfg = self.config
        with self.lock:
            sim = self.suite.sim
            rgb, depth = sim.render(camera_name=cam, width=cfg.camera_width, height=cfg.camera_height, depth=True)
            # MuJoCo renders bottom-up and returns a normalized depth buffer.
            rgb = np.ascontiguousarray(rgb[::-1])
            depth = get_real_depth_map(sim, depth[::-1]).reshape(rgb.shape[:2]).astype(np.float32)
            intrinsics = get_camera_intrinsic_matrix(sim, cam, cfg.camera_height, cfg.camera_width)
            cam_to_world = get_camera_extrinsic_matrix(sim, cam)
        return RGBD(rgb, depth, intrinsics, cam_to_world)

    def wait_for_next_tick(self, dt: float) -> None:
        steps = max(1, round(dt * self.config.control_freq))
        for _ in range(steps):
            with self.lock:
                self.suite.step(self._action())
                self.contact_ticks += self._base_in_contact()

    # -- RoboCasa-specific helpers (used by tests and scripts, not by the robot layer) --

    def teleport_base(self, pose: Pose2D) -> None:
        """Place the base at `pose` instantly, at rest."""
        with self.lock:
            current = self._base_pose()
            delta = np.array([pose.x - current.x, pose.y - current.y])
            for joint, qpos in zip(self._slide_joints, self._base_qpos[:2]):
                self.data.qpos[qpos] += float(self.data.xaxis[joint][:2] @ delta)
            self.data.qpos[self._base_qpos[2]] += wrap_angle(pose.yaw - current.yaw)
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self._command_world = np.zeros(3)

    def close(self) -> None:
        with self.lock:
            self.suite.close()

    # -- internals ------------------------------------------------------------

    def _base_pose(self) -> Pose2D:
        anchor = self.data.xanchor[self._yaw_joint]
        rotation = self.data.xmat[self._base_body]
        return Pose2D(float(anchor[0]), float(anchor[1]), math.atan2(rotation[3], rotation[0]))

    def _base_in_contact(self) -> bool:
        """True if any part of the robot touches something other than the floor."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.dist > 0:
                continue
            first, second = contact.geom1 in self.robot_geoms, contact.geom2 in self.robot_geoms
            if first == second:
                continue
            other = contact.geom2 if first else contact.geom1
            if other not in self._floor_geoms:
                return True
        return False

    def _action(self) -> np.ndarray:
        """The robosuite action that realizes the commanded world-frame base velocity."""
        vx, vy, vyaw = self._command_world
        # Velocity along each slide joint's (world-frame) axis, then the yaw rate.
        target = np.array(
            [self.data.xaxis[j][0] * vx + self.data.xaxis[j][1] * vy for j in self._slide_joints] + [vyaw]
        )
        ctrl = target + self._friction_offset * np.tanh(target / _FRICTION_FADE)
        ctrl = np.clip(ctrl / self._ctrl_max, -1.0, 1.0)
        # robosuite's base controller maps its input into the joint frame; undo
        # that mapping up front so the joints receive exactly `ctrl`.
        base_action = np.append(np.linalg.solve(self._controller_mixing(), ctrl[:2]), ctrl[2])
        # base_mode=1: the arm holds its pose relative to the moving base.
        return self._controller.create_action_vector({"base": base_action, "base_mode": 1})

    def _controller_mixing(self) -> np.ndarray:
        """The 2x2 matrix robosuite's base controller applies to its xy input
        (see LegacyMobileBaseJointVelocityController.set_goal)."""
        controller = self._base_controller
        _, orientation = controller.get_base_pose()
        theta = T.mat2euler(orientation)[2] - T.mat2euler(controller.init_ori)[2]
        c, s = math.cos(theta), math.sin(theta)
        if controller._check_forward_joint_reversed():
            return np.array([[c, s], [-s, c]]) @ np.array([[0.0, 1.0], [1.0, 0.0]])
        return np.array([[c, -s], [s, c]])
