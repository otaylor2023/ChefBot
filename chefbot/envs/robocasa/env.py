"""RoboCasaEnv: the Env interface on a RoboCasa (robosuite + MuJoCo) kitchen.

Low-level only: it accepts base velocity commands, steps physics, and reports
state. Planning and control live in chefbot/robot.
"""

from __future__ import annotations

import math
import threading
from concurrent.futures import ThreadPoolExecutor

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
from chefbot.envs.robocasa.gripper import PAD_GEOMS, register_gripper
from chefbot.envs.robocasa.spawn import TASK_NAME, place_objects
from chefbot.interface.env import Env
from chefbot.interface.types import RGBD, EnvState, Pose2D, wrap_angle

_ROBOT_PREFIXES = ("robot0_", "mobilebase0_", "gripper0_")
_BASE_BODY = "mobilebase0_base"
_BASE_JOINTS = ("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side", "mobilebase0_joint_mobile_yaw")
_BASE_ACTUATORS = ("mobilebase0_actuator_mobile_forward", "mobilebase0_actuator_mobile_side", "mobilebase0_actuator_mobile_yaw")
_ARM_JOINTS = tuple(f"robot0_joint{i}" for i in range(1, 8))
_ARM_PART, _GRIPPER_PART = "right", "right_gripper"
TOOL_BIT = 1 << 3  # collision bit a locked utensil keeps: only geoms with it in their conaffinity touch it
LOCKED_MASS = 0.5  # kg: a locked object is at least this heavy while held (well over what is in it)
CARRY_TILT = 70.0  # deg: a locked container held still tipped further lets go of what is in it: it falls out
# past the lip (on a gentler slope the thin pan floors let food sink through; while it still turns,
# its floor would sweep what is falling out sideways)
CARRY_STILL = 0.05  # rad/s: turning slower than this is held still
CARRY_ABOVE = 0.06  # m above a container's rim that still counts as in it
_GRIPPER_PREFIX = "gripper0_right_"
# Arm links that count for `arm_in_contact`. The first two links sit on the
# base and never reach anything; the gripper body counts, its fingers do not.
_ARM_CONTACT_BODIES = tuple(f"robot0_link{i}" for i in range(2, 8)) + ("robot0_right_hand",)
_GRIPPER_CONTACT_GEOMS = ("base_mount_collision", "base_collision")
# Pad geoms are 8 mm thick boxes: the gap between their faces is this much
# less than the distance between their centers.
_PAD_THICKNESS = 0.008
# Below this commanded speed the friction feedforward fades out, so a zero
# command means zero actuator effort. It has to be small: with the no-slip
# solver on, joint friction is a hard threshold, and a slow final correction
# only moves the base if nearly all of the feedforward is applied.
_FRICTION_FADE = np.array([0.0005, 0.0005, 0.001])


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
        # All camera rendering happens on this one thread: on macOS, MuJoCo's GL
        # context stays bound to the thread that last used it, and the robot
        # layer (motion thread), the detector and recorders all ask for images.
        self._render_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render") if render else None
        controller_config = load_composite_controller_config(robot=cfg.robot)
        # The arm takes absolute joint position targets; robosuite turns them
        # into torques (PD plus gravity compensation) at the physics rate.
        controller_config["body_parts"][_ARM_PART] = {
            "type": "JOINT_POSITION",
            "input_type": "absolute",
            "input_max": 1,
            "input_min": -1,
            "output_max": 1,
            "output_min": -1,
            "kp": cfg.arm_kp,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 1000],
            "damping_ratio_limits": [0, 10],
            "qpos_limits": None,
            "interpolation": None,
            "ramp_ratio": 0.2,
            "gripper": {"type": "GRIP"},
        }
        # Every sim access goes through this lock: the motion thread steps
        # physics while other threads read poses and run detection.
        self.lock = threading.RLock()
        self.suite = robosuite.make(
            env_name=TASK_NAME,
            robots=cfg.robot,
            gripper_types=register_gripper(),
            controller_configs=controller_config,
            chef_objects=cfg.objects,
            layout_ids=cfg.layout,
            style_ids=cfg.style,
            # aigen has foods the others lack (sausage, bacon, steak); every object we spawn is pinned.
            obj_registries=("objaverse", "lightwheel", "aigen"),
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
        model.opt.noslip_iterations = cfg.noslip_iterations

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
        self.contact_ticks = 0  # control ticks that ended with the base or arm touching something
        self.last_contact: list[tuple[str, str]] = []  # (robot geom, other geom) at the last such tick

        self.robot_geoms = frozenset(
            g for g in range(model.ngeom) if self.is_robot_body(int(model.geom_bodyid[g]))
        )
        self._floor_geoms = frozenset(
            g for g in range(model.ngeom) if self.root_body_name(int(model.geom_bodyid[g])).startswith("floor")
        )
        self._base_geoms = frozenset(
            g for g in self.robot_geoms if model.body(int(model.geom_bodyid[g])).name.startswith("mobilebase0_")
        )
        # The base rides on slide joints, not on its feet: with the no-slip solver
        # on, foot-on-floor friction would hold it back at low speed. Its geoms
        # take priority in every contact and bring no friction.
        for geom in self._base_geoms:
            model.geom_priority[geom] = 1
            model.geom_friction[geom] = 0.0
        arm_bodies = {model.body(name).id for name in _ARM_CONTACT_BODIES}
        self._arm_geoms = frozenset(
            g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in arm_bodies
        ) | {model.geom(_GRIPPER_PREFIX + name).id for name in _GRIPPER_CONTACT_GEOMS}
        self._pad_geoms = {
            side: frozenset(model.geom(_GRIPPER_PREFIX + name).id for name in names)
            for side, names in PAD_GEOMS.items()
        }

        arm_joints = [model.joint(name).id for name in _ARM_JOINTS]
        self._arm_qpos = [int(model.jnt_qposadr[j]) for j in arm_joints]
        self._arm_dofs = [int(model.jnt_dofadr[j]) for j in arm_joints]
        self.arm_joint_limits = model.jnt_range[arm_joints].copy()
        self.grip_site = model.site(_GRIPPER_PREFIX + "grip_site").id
        self._arm_target = self.data.qpos[self._arm_qpos].copy()  # hold the starting pose
        self._gripper_closed = False
        # Objects with an adaptive handle (utensils): once the closed gripper has
        # them between both pads they lock to it, as a form-fit or magnetic
        # handle would, until it opens. Root body ids; the lock is (body, pose in
        # the grip frame).
        self.lockable: set[int] = set()
        self._locked: tuple[int, np.ndarray] | None = None
        self._lock_scale = 1.0
        self._carried: list[tuple[int, np.ndarray]] = []  # (body, pose in the locked container)
        self._carried_up = np.array([0.0, 0.0, 1.0])  # world up, in the container's frame when locked
        # While locked it touches only geoms with TOOL_BIT in their conaffinity
        # (the food): held rigidly it would shove the pan along by its rim.
        # Except these (cookware: a pan stands on burners and holds the food).
        self.keeps_contacts: set[int] = set()
        self._unlocked_bits: dict[int, tuple[int, int]] = {}
        for geom in set().union(*self._pad_geoms.values()):  # the pads keep their grip on it
            model.geom_conaffinity[geom] |= TOOL_BIT

        place_objects(self)
        # The scene as spawned, for reset_scene(): everything that is not a robot joint.
        robot_joints = [j for j in range(model.njnt) if self.is_robot_body(int(model.jnt_bodyid[j]))]
        robot_qpos = {int(model.jnt_qposadr[j]) for j in robot_joints}  # robot joints are all 1-DoF
        self._scene_qpos = np.array([i for i in range(model.nq) if i not in robot_qpos])
        self._scene_start = self.data.qpos[self._scene_qpos].copy()

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
            base_contact, arm_contact = self._contacts()
            return EnvState(
                pose,
                base_velocity,
                base_contact,
                float(self.data.time),
                arm_q=self.data.qpos[self._arm_qpos].copy(),
                arm_qd=qvel[self._arm_dofs].copy(),
                gripper_width=self._gripper_width(),
                gripper_closed_cmd=self._gripper_closed,
                arm_in_contact=arm_contact,
            )

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
        q = np.asarray(q, dtype=float).reshape(7)
        with self.lock:
            self._arm_target = np.clip(q, self.arm_joint_limits[:, 0], self.arm_joint_limits[:, 1])

    def set_gripper(self, closed: bool) -> None:
        with self.lock:
            self._gripper_closed = bool(closed)

    def get_camera_rgbd(self, cam: str) -> RGBD:
        """Do not call with `lock` held: the render thread takes it."""
        if not self._render:
            raise RuntimeError("RoboCasaEnv was created with render=False; pass render=True for cameras")
        return self._render_thread.submit(self._render_rgbd, cam).result()

    def _render_rgbd(self, cam: str) -> RGBD:
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
                self._hold_locked()
                if any(self._contacts()):
                    self.contact_ticks += 1
                    self.last_contact = self._contact_names()

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

    def set_arm_joints(self, q: np.ndarray) -> None:
        """Put the arm at `q` instantly, at rest, and hold it there."""
        with self.lock:
            self.data.qpos[self._arm_qpos] = np.asarray(q, dtype=float)
            self.data.qvel[self._arm_dofs] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self._arm_target = self.data.qpos[self._arm_qpos].copy()

    def reset_scene(self) -> None:
        """Put every object and every door back where it was at startup, and open
        the gripper. The robot stays where it is."""
        with self.lock:
            self.data.qpos[self._scene_qpos] = self._scene_start
            self.data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self._gripper_closed = False
            self._set_locked(None)
            self._command_world = np.zeros(3)
        width = None
        for tick in range(4 * self.config.control_freq):  # until the gripper has opened (from a closed grip, too)
            self.wait_for_next_tick(1.0 / self.config.control_freq)
            before, width = width, self.get_state().gripper_width
            if tick >= 10 and abs(width - before) < 1e-4:
                break

    def tcp_pose(self) -> np.ndarray:
        """World pose (4x4) of the gripper's grip site."""
        with self.lock:
            pose = np.eye(4)
            pose[:3, :3] = self.data.site_xmat[self.grip_site].reshape(3, 3)
            pose[:3, 3] = self.data.site_xpos[self.grip_site]
            return pose

    def grasped_bodies(self) -> set[int]:
        """Root bodies touching both fingertip pads right now."""
        with self.lock:
            return self._grasped_roots()

    def close(self) -> None:
        if self._render_thread is not None:
            self._render_thread.shutdown()
        with self.lock:
            self.suite.close()

    # -- internals ------------------------------------------------------------

    def _base_pose(self) -> Pose2D:
        anchor = self.data.xanchor[self._yaw_joint]
        rotation = self.data.xmat[self._base_body]
        return Pose2D(float(anchor[0]), float(anchor[1]), math.atan2(rotation[3], rotation[0]))

    def _touching(self) -> list[tuple[int, int]]:
        """(geom, geom) for every pair in contact."""
        contacts = self.data.contact[: self.data.ncon]
        return [(int(c.geom1), int(c.geom2)) for c in contacts if c.dist <= 0]

    def _root_of_geom(self, geom: int) -> int:
        return int(self.model.body_rootid[self.model.geom_bodyid[geom]])

    def _grip_pose(self) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = self.data.site_xmat[self.grip_site].reshape(3, 3)
        pose[:3, 3] = self.data.site_xpos[self.grip_site]
        return pose

    def _body_pose(self, body: int) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = self.data.xmat[body].reshape(3, 3)
        pose[:3, 3] = self.data.xpos[body]
        return pose

    def _hold_locked(self) -> None:
        """Keep a locked object where the grip holds it; lock or release with the gripper."""
        if not self._gripper_closed:
            if self._locked is not None:
                self._set_locked(None)
            return
        if self._locked is None:
            grasped = self._grasped_roots() & self.lockable
            if grasped:
                body = min(grasped)
                self._set_locked((body, np.linalg.inv(self._grip_pose()) @ self._body_pose(body)))
            return
        body, in_grip = self._locked
        pose = self._grip_pose() @ in_grip
        grip = np.zeros(6)  # angular, linear (world)
        if body in self.keeps_contacts:
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self.grip_site, grip, 0)
        # (A utensil stays put between ticks: moving, it would shove the food out of the pan.)
        self._drive(body, pose, grip)
        for other, in_body in self._carried:
            self._drive(other, pose @ in_body, grip)  # also on the last tick: let go moving with the pan
        tipped = (pose[:3, :3] @ self._carried_up)[2] < math.cos(math.radians(CARRY_TILT))
        if self._carried and tipped and np.linalg.norm(grip[:3]) < CARRY_STILL:
            self._carry(None)  # held tipped: what is in it falls out (pouring)
        mujoco.mj_kinematics(self.model, self.data)

    def _drive(self, body: int, pose: np.ndarray, grip: np.ndarray) -> None:
        """Put a free body at `pose`, moving with the grip until the next tick (not
        falling and snapping back up, which tosses what is in a pan)."""
        joint = int(self.model.body_jntadr[body])
        if joint < 0 or self.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            return
        qpos, dof = int(self.model.jnt_qposadr[joint]), int(self.model.jnt_dofadr[joint])
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, pose[:3, :3].reshape(-1))
        self.data.qpos[qpos : qpos + 7] = np.concatenate([pose[:3, 3], quat])
        spin = grip[:3]
        self.data.qvel[dof : dof + 3] = grip[3:] + np.cross(spin, pose[:3, 3] - self.data.site_xpos[self.grip_site])
        self.data.qvel[dof + 3 : dof + 6] = pose[:3, :3].T @ spin  # a free joint's spin is in its own frame

    def _set_locked(self, lock: tuple[int, np.ndarray] | None) -> None:
        model = self.model
        self._carry(None)
        for geom, (contype, conaffinity) in self._unlocked_bits.items():
            model.geom_contype[geom], model.geom_conaffinity[geom] = contype, conaffinity
        self._unlocked_bits = {}
        if self._locked is not None:
            bodies = model.body_rootid == self._locked[0]
            model.body_gravcomp[bodies] = 0.0
            model.body_mass[bodies] /= self._lock_scale
            model.body_inertia[bodies] /= self._lock_scale
            self._lock_scale = 1.0
        self._locked = lock
        if lock is None:
            return
        if lock[0] in self.keeps_contacts:
            # Gravity off and heavy, as if driven by the arm: what is in a light pan
            # would otherwise weigh it down between ticks and the lock would snap its
            # thin floor up through the food.
            bodies = model.body_rootid == lock[0]
            self._lock_scale = max(1.0, LOCKED_MASS / float(model.body_subtreemass[lock[0]]))
            model.body_gravcomp[bodies] = 1.0
            model.body_mass[bodies] *= self._lock_scale
            model.body_inertia[bodies] *= self._lock_scale
            self._carry(lock[0])
            return
        for geom in np.flatnonzero(model.body_rootid[model.geom_bodyid] == lock[0]):
            geom = int(geom)
            if model.geom_contype[geom] or model.geom_conaffinity[geom]:
                self._unlocked_bits[geom] = (int(model.geom_contype[geom]), int(model.geom_conaffinity[geom]))
                model.geom_contype[geom] = model.geom_conaffinity[geom] = TOOL_BIT

    def _carry(self, container: int | None) -> None:
        """What is in a locked container rides along with it while it is held about
        level (a level pan keeps its food; the sim's thin pan floors let food sink
        through when lifted); dropped from it once held still tipped past CARRY_TILT.
        None: let go of what is carried."""
        model = self.model
        for other, _ in self._carried:
            model.body_gravcomp[model.body_rootid == other] = 0.0
        self._carried = []
        if container is None:
            return
        lo, hi = self._world_box(container)
        pose = self._body_pose(container)
        for other in range(1, model.nbody):
            joint = int(model.body_jntadr[other])
            if (model.body_parentid[other] != 0 or other == container or joint < 0 or self.is_robot_body(other)
                    or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE):  # fmt: skip
                continue
            middle = self.data.subtree_com[other]
            if np.all(middle[:2] >= lo[:2]) and np.all(middle[:2] <= hi[:2]) and lo[2] <= middle[2] <= hi[2] + CARRY_ABOVE:
                self._carried.append((other, np.linalg.inv(pose) @ self._body_pose(other)))
                model.body_gravcomp[model.body_rootid == other] = 1.0
        self._carried_up = pose[:3, :3].T @ [0.0, 0.0, 1.0]

    def _world_box(self, root: int) -> tuple[np.ndarray, np.ndarray]:
        """World min / max corners of a body's collision geometry."""
        model, data = self.model, self.data
        corners = []
        for geom in np.flatnonzero(model.body_rootid[model.geom_bodyid] == root):
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            center, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
            local = center + half * (2 * np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) - 1)
            corners.append(local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
        points = np.concatenate(corners) if corners else data.xpos[root][None]
        return points.min(axis=0), points.max(axis=0)

    def _grasped_roots(self, pairs: list[tuple[int, int]] | None = None) -> set[int]:
        locked = {self._locked[0]} if self._locked is not None else set()
        pairs = self._touching() if pairs is None else pairs
        touched = {"left": set(), "right": set()}
        for a, b in pairs:
            for side, pads in self._pad_geoms.items():
                if a in pads and b not in self.robot_geoms:
                    touched[side].add(self._root_of_geom(b))
                elif b in pads and a not in self.robot_geoms:
                    touched[side].add(self._root_of_geom(a))
        return (touched["left"] & touched["right"]) | locked

    def _contacts(self) -> tuple[bool, bool]:
        """(base_in_contact, arm_in_contact).

        Base: the mobile base touches something other than the floor.
        Arm: links 2-7, the hand or the gripper body touch the world or the
        mobile base. Finger contacts never count, and neither does contact
        with an object held between the pads.
        """
        pairs = self._touching()
        grasped = self._grasped_roots(pairs)
        base = arm = False
        for a, b in pairs:
            for own, other in ((a, b), (b, a)):
                if own in self._base_geoms and other not in self.robot_geoms and other not in self._floor_geoms:
                    base = True
                if own in self._arm_geoms and other not in self._arm_geoms:
                    if other in self._base_geoms:
                        arm = True
                    elif other not in self.robot_geoms and self._root_of_geom(other) not in grasped:
                        arm = True
        return base, arm

    def _contact_names(self) -> list[tuple[str, str]]:
        """(robot geom, other geom) names behind `base_in_contact` / `arm_in_contact`."""
        pairs = self._touching()
        grasped = self._grasped_roots(pairs)
        names = []
        for a, b in pairs:
            for own, other in ((a, b), (b, a)):
                base = own in self._base_geoms and other not in self.robot_geoms and other not in self._floor_geoms
                arm = own in self._arm_geoms and other not in self._arm_geoms and (
                    other in self._base_geoms
                    or (other not in self.robot_geoms and self._root_of_geom(other) not in grasped)
                )
                if base or arm:
                    names.append((self.model.geom(own).name, self.model.geom(other).name))
        return names

    def _gripper_width(self) -> float:
        left, right = (next(iter(self._pad_geoms[side])) for side in ("left", "right"))
        gap = float(np.linalg.norm(self.data.geom_xpos[left] - self.data.geom_xpos[right]))
        return max(0.0, gap - _PAD_THICKNESS)

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
        return self._controller.create_action_vector(
            {
                "base": base_action,
                _ARM_PART: self._arm_target,
                _GRIPPER_PART: np.array([1.0 if self._gripper_closed else -1.0]),
            }
        )

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
