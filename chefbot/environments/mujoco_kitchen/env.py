"""A kitchen in plain MuJoCo: an exported RoboCasa kitchen (export.py), the TidyBot++ with the FR3 and Franka Hand
(robot.py), and any extra free objects (a generated pair of tongs), as one model. Implements ChefBot's Env.

    env = MujocoKitchenEnv(kitchen_dir, extras={"tongs": "path/to/model.xml"})
    env.set_base_target(pose, max_vel) / set_arm_joint_targets(q) / set_gripper(closure 0..1) / wait_for_next_tick(dt)
    env.get_state(), env.get_camera_rgbd("tb/pole"), env.render(cam, w, h)
    env.body_root(name) -> body id; env.geoms_of(root) -> geom ids; env.contact_forces() -> {root body: N}

Physics: RoboCasa's options (2 ms steps, elliptic cones, impratio 20); objects get real masses (objects.py). The
base's position servos follow a reference that moves toward the target within the speed limits (the real base's
trajectory generator); the reference never leads the base by more than BASE_LEAD, so a blocked base pushes gently.
Hooks in `physics_hooks` run after every physics step (the task's fragile food watches its contact forces there).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from chefbot.environments.mujoco_kitchen.objects import DEFAULT, MASS
from chefbot.environments.mujoco_kitchen.robot import _absolute_files, robot_spec
from chefbot.interface.env import Env
from chefbot.interface.types import RGBD, EnvState, Pose2D

ROBOT = "tb/"
ARM_JOINTS = [f"{ROBOT}fr3/fr3_joint{i}" for i in range(1, 8)]
FINGERS = [f"{ROBOT}fr3/hand/finger_joint1", f"{ROBOT}fr3/hand/finger_joint2"]
FINGER_BODIES = (f"{ROBOT}fr3/hand/left_finger", f"{ROBOT}fr3/hand/right_finger")
BASE_LEAD = (0.05, 0.10)  # m, rad: how far the base's reference may lead it
CAMERA_SIZE = (640, 480)
VISUAL = mujoco.MjvOption()  # cameras see the visual geoms (groups 1, 2), not the collision ones (0, 3)
VISUAL.geomgroup[:] = [0, 1, 1, 0, 0, 0]


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class MujocoKitchenEnv(Env):
    def __init__(self, kitchen_dir: Path, extras: dict[str, Path] | None = None, render: bool = True):
        self.kitchen_dir = Path(kitchen_dir)
        self.meta = json.loads((self.kitchen_dir / "meta.json").read_text())
        spec = mujoco.MjSpec.from_file(str(self.kitchen_dir / "kitchen.xml"))
        spec.compiler.degree = False
        self._real_masses(spec)
        spec.worldbody.add_frame().attach_body(robot_spec().body("base_link"), ROBOT, "")
        self.extras = {}
        for name, path in (extras or {}).items():
            child = mujoco.MjSpec.from_file(str(path))
            _absolute_files(child, Path(path).parent / (child.meshdir or ""))
            body = spec.worldbody.add_frame(pos=[0, 0, -5.0 - len(self.extras)]).attach_body(child.body("object"), f"{name}/", "")
            body.add_freejoint(name=f"{name}/free")
            self.extras[name] = f"{name}/object"
        w, h = CAMERA_SIZE
        spec.visual.global_.offwidth, spec.visual.global_.offheight = max(w, 1280), max(h, 960)
        spec.visual.map.znear = 0.01
        self.model = spec.compile()
        # RoboCasa's collision geoms are invisible (alpha 0), and mj_ray skips invisible geoms: the scene's rays must
        # see them. Cameras render only the visual groups (VISUAL), so this changes no image.
        collision = np.isin(self.model.geom_group, [0, 3]) & (self.model.geom_rgba[:, 3] == 0)
        self.model.geom_rgba[collision, 3] = 1.0
        self.data = mujoco.MjData(self.model)
        self.physics_hooks: list = []
        m = self.model
        self._base_qpos = [m.jnt_qposadr[m.joint(f"{ROBOT}joint_{a}").id] for a in ("x", "y", "th")]
        self._base_dof = [m.jnt_dofadr[m.joint(f"{ROBOT}joint_{a}").id] for a in ("x", "y", "th")]
        self._base_act = [m.actuator(f"{ROBOT}joint_{a}").id for a in ("x", "y", "th")]
        self._arm_qpos = [m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS]
        self._arm_dof = [m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS]
        self._arm_act = [m.actuator(j).id for j in ARM_JOINTS]
        self._finger_qpos = [m.jnt_qposadr[m.joint(j).id] for j in FINGERS]
        self._grip_act = m.actuator(f"{ROBOT}fr3/hand/actuator8").id
        self.robot_root = m.body(f"{ROBOT}base_link").id
        self.robot_geoms = frozenset(int(g) for g in np.flatnonzero(m.body_rootid[m.geom_bodyid] == self.robot_root))
        self.base_geoms = frozenset(int(g) for g in np.flatnonzero(m.geom_bodyid == self.robot_root))
        fingers = {m.body(b).id for b in FINGER_BODIES}
        self.finger_geoms = frozenset(int(g) for g in np.flatnonzero(np.isin(m.geom_bodyid, list(fingers))))
        self.hand_geoms = frozenset(g for g in self.robot_geoms if self._under(int(m.geom_bodyid[g]), m.body(f"{ROBOT}fr3/hand/hand").id))
        self.floor_geoms = frozenset(m.geom(n).id for n in self.meta["floor_geoms"])
        self._renderers: dict[tuple[int, int, bool], mujoco.Renderer] = {}
        self._render = render
        self._clamp_masses()
        self.reset_scene()

    # -- model ---------------------------------------------------------------------

    def _real_masses(self, spec: mujoco.MjSpec) -> None:
        """Collision geoms of each object take its category's density, visual geoms none (objects.py)."""
        self._mass_range = {}
        for o in self.meta["objects"]:
            density, lo, hi = MASS.get(o["category"], DEFAULT)
            body = spec.body(o["body"])
            for g in body.geoms:
                visual = g.contype == 0 and g.conaffinity == 0
                g.density = 0.0 if visual else float(density)
            self._mass_range[o["body"]] = (lo, hi)

    def _clamp_masses(self) -> None:
        m = self.model
        for name, (lo, hi) in self._mass_range.items():
            b = m.body(name).id
            ids = [i for i in range(m.nbody) if m.body_rootid[i] == b]
            total = float(sum(m.body_mass[i] for i in ids))
            k = float(np.clip(total, lo, hi)) / max(total, 1e-6)
            for i in ids:
                m.body_mass[i] *= k
                m.body_inertia[i] *= k
        mujoco.mj_setConst(m, self.data)

    def _under(self, body: int, root: int) -> bool:
        while body > 0:
            if body == root:
                return True
            body = int(self.model.body_parentid[body])
        return False

    def body_root(self, name: str) -> int:
        return int(self.model.body(name).id)

    def geoms_of(self, root: int, collision_only: bool = True) -> frozenset[int]:
        m = self.model
        ids = np.flatnonzero(m.body_rootid[m.geom_bodyid] == root)
        return frozenset(int(g) for g in ids if not collision_only or (m.geom_contype[g] or m.geom_conaffinity[g]))

    def free_qpos(self, root: int) -> int:
        return int(self.model.jnt_qposadr[self.model.body_jntadr[root]])

    # -- episode ---------------------------------------------------------------------

    def reset_scene(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self._base_ref = np.zeros(3)
        self._base_goal = None  # (pose, max_vel) or ("vel", v)
        self._base_vel_cmd = np.zeros(3)
        self.gripper_cmd = 0.0
        self.data.ctrl[self._grip_act] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def teleport_base(self, pose: Pose2D) -> None:
        d = self.data
        d.qpos[self._base_qpos] = [pose.x, pose.y, pose.yaw]
        d.qvel[self._base_dof] = 0.0
        self._base_ref = np.array([pose.x, pose.y, pose.yaw])
        self._base_goal = None
        self._base_vel_cmd[:] = 0.0
        d.ctrl[self._base_act] = self._base_ref
        mujoco.mj_forward(self.model, d)

    def set_arm_joints(self, q) -> None:
        d = self.data
        d.qpos[self._arm_qpos] = q
        d.qvel[self._arm_dof] = 0.0
        d.ctrl[self._arm_act] = q
        mujoco.mj_forward(self.model, d)

    # -- Env -------------------------------------------------------------------------

    def get_state(self) -> EnvState:
        d = self.data
        x, y, th = d.qpos[self._base_qpos]
        return EnvState(base_pose=Pose2D(float(x), float(y), float(th)), base_velocity=d.qvel[self._base_dof].copy(),
                        base_in_contact=self._base_contact(), time=float(d.time), arm_q=d.qpos[self._arm_qpos].copy(),
                        arm_qd=d.qvel[self._arm_dof].copy(), gripper_width=self.gripper_width(),
                        gripper_closed_cmd=self.gripper_cmd >= 0.5, arm_in_contact=self._arm_contact())  # fmt: skip

    def gripper_width(self) -> float:
        return float(np.sum(self.data.qpos[self._finger_qpos]))

    def set_base_velocity(self, vx: float, vy: float, vyaw: float, frame: str = "world") -> None:
        v = np.array([vx, vy, vyaw], float)
        if frame == "base":
            th = float(self.data.qpos[self._base_qpos[2]])
            c, s = math.cos(th), math.sin(th)
            v[:2] = [c * vx - s * vy, s * vx + c * vy]
        self._base_goal = None
        self._base_vel_cmd = v

    def set_base_target(self, pose: Pose2D, max_vel) -> bool:
        self._base_goal = (np.array([pose.x, pose.y, pose.yaw], float), np.asarray(max_vel, float))
        self._base_vel_cmd[:] = 0.0
        return True

    def set_arm_joint_targets(self, q) -> None:
        self.data.ctrl[self._arm_act] = np.asarray(q, float)

    def set_gripper(self, closed) -> None:
        """Closure 0 (open) .. 1 (closed); a bool is fully open or closed."""
        self.gripper_cmd = float(np.clip(float(closed), 0.0, 1.0))
        self.data.ctrl[self._grip_act] = self.gripper_cmd

    def wait_for_next_tick(self, dt: float) -> None:
        m, d = self.model, self.data
        for _ in range(max(1, round(dt / m.opt.timestep))):
            self._advance_base_ref(m.opt.timestep)
            mujoco.mj_step(m, d)
            for hook in self.physics_hooks:
                hook(self)

    def _advance_base_ref(self, h: float) -> None:
        d = self.data
        actual = d.qpos[self._base_qpos]
        if self._base_goal is not None:
            goal, vmax = self._base_goal
            err = goal - self._base_ref
            err[2] = wrap(err[2])
            lin = np.linalg.norm(err[:2])
            step = np.zeros(3)
            if lin > 1e-9:
                step[:2] = err[:2] * min(1.0, vmax[0] * h / lin)
            step[2] = float(np.clip(err[2], -vmax[2] * h, vmax[2] * h))
            self._base_ref = self._base_ref + step
        else:
            self._base_ref = self._base_ref + self._base_vel_cmd * h
        lead = self._base_ref - actual
        lead[2] = wrap(lead[2])
        n = np.linalg.norm(lead[:2])
        if n > BASE_LEAD[0]:
            lead[:2] *= BASE_LEAD[0] / n
        lead[2] = float(np.clip(lead[2], -BASE_LEAD[1], BASE_LEAD[1]))
        self._base_ref = actual + lead
        d.ctrl[self._base_act] = self._base_ref

    # -- contacts --------------------------------------------------------------------

    def contacts(self):
        """(geom1, geom2, normal force N) for every active contact."""
        m, d = self.model, self.data
        f = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.efc_address < 0:
                continue
            mujoco.mj_contactForce(m, d, i, f)
            yield int(c.geom1), int(c.geom2), float(abs(f[0]))

    def _touching(self) -> list[tuple[int, int]]:
        return [(int(d.geom1), int(d.geom2)) for d in self.data.contact[: self.data.ncon]]

    def _base_contact(self) -> bool:
        for a, b in self._touching():
            if (a in self.base_geoms) != (b in self.base_geoms):
                other = b if a in self.base_geoms else a
                if other not in self.floor_geoms and other not in self.robot_geoms:
                    return True
        return False

    def _arm_contact(self) -> bool:
        arm = self.robot_geoms - self.base_geoms - self.finger_geoms
        for a, b in self._touching():
            if (a in arm) != (b in arm) and not ((a in self.robot_geoms) and (b in self.robot_geoms)):
                return True
        return False

    # -- cameras ---------------------------------------------------------------------

    def _renderer(self, width: int, height: int, depth: bool) -> mujoco.Renderer:
        key = (width, height, depth)
        if key not in self._renderers:
            r = mujoco.Renderer(self.model, height, width)
            if depth:
                r.enable_depth_rendering()
            self._renderers[key] = r
        return self._renderers[key]

    def render(self, cam, width: int, height: int) -> np.ndarray:
        """RGB from a named camera or an MjvCamera."""
        r = self._renderer(width, height, False)
        r.update_scene(self.data, cam, VISUAL)
        return r.render().copy()

    def get_camera_rgbd(self, cam: str) -> RGBD:
        if not self._render:
            raise RuntimeError("MujocoKitchenEnv was created with render=False")
        w, h = CAMERA_SIZE
        rgb = self.render(cam, w, h)
        r = self._renderer(w, h, True)
        r.update_scene(self.data, cam, VISUAL)
        depth = r.render().astype(np.float32)
        m, d = self.model, self.data
        i = m.camera(cam).id
        f = 0.5 * h / math.tan(math.radians(float(m.cam_fovy[i])) / 2)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        T = np.eye(4)
        T[:3, :3] = d.cam_xmat[i].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = d.cam_xpos[i]
        return RGBD(rgb, depth, K, T)

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers = {}

