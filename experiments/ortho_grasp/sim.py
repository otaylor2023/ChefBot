"""The RoboCasa kitchen for the experiment: the robot parked at a fixed pose
facing a counter, every cup and distractor spawned once, and a scene applied
by moving them (unused ones are parked out of the way, floating).

Uses chefbot read-only: the env (subclassed here only to hold the gripper
part-open), the robot stack for IK, collision and arm control.
"""

from __future__ import annotations

import contextlib
import dataclasses
import threading
from pathlib import Path

import mujoco
import numpy as np

from chefbot.envs.robocasa.build_map import ensure_map
from chefbot.envs.robocasa.config import RoboCasaConfig
from chefbot.envs.robocasa.detector import scene_objects
from chefbot.envs.robocasa.env import RoboCasaEnv
from chefbot.envs.robocasa.objects import object_points
from chefbot.envs.robocasa.serve import wire_stack
from chefbot.interface.types import Pose2D
from chefbot.robot.config import RobotConfig

from experiments.ortho_grasp.frames import to_robot, to_world

PARK_Z = -3.0  # m (world): unused objects float here, out of every view


class ExperimentEnv(RoboCasaEnv):
    """RoboCasaEnv whose gripper can also be held where it is (part open)."""

    gripper_hold = False

    def _action(self) -> np.ndarray:
        action = super()._action()
        if self.gripper_hold:
            start, end = self._controller._action_split_indexes[_gripper_part(self)]
            action[start:end] = 0.0  # the 2F-85's action is incremental: 0 keeps the fingers where they are
        return action


def _gripper_part(env: RoboCasaEnv) -> str:
    return next(name for name in env._controller._action_split_indexes if "gripper" in name)


class World:
    """One spawned kitchen. `objects` maps a key ("cup:cup_2", "apple") to its body id."""

    def __init__(self, config: dict, render: bool = True):
        self.config = config
        b = config["base_pose"]
        self.base = Pose2D(float(b[0]), float(b[1]), float(np.radians(b[2])))
        cfg = RoboCasaConfig.load(layout=int(config["layout"]), style=int(config["layout"]))
        specs = [{"category": "cup", "model": f"cup/{m}", "surface": "counter"} for m in config["cup_models"]]
        specs += [{"category": d["category"], "model": d["model"], "surface": "counter"} for d in config["distractors"]]
        cfg = dataclasses.replace(cfg, objects=tuple(specs), camera_width=int(config["camera"][0]), camera_height=int(config["camera"][1]))
        self.env = ExperimentEnv(cfg, render=render)
        spawned = scene_objects(self.env)  # (sim name, label, body), in spawn order
        if len(spawned) != len(specs):
            raise RuntimeError(f"spawned {len(spawned)} objects, expected {len(specs)}")
        keys = [f"cup:{m}" for m in config["cup_models"]] + [d["category"] for d in config["distractors"]]
        m = self.env.model
        self.objects = {key: int(m.body_rootid[body]) for key, (_, _, body) in zip(keys, spawned)}
        # Each object's lowest point below its body origin, upright (to stand it on the counter).
        self._below = {key: float(self.env.data.xpos[b][2] - object_points(self.env, b)[:, 2].min()) for key, b in self.objects.items()}
        self.counter_z = float(config["counter_z"])
        map_dir = ensure_map(self.env, Path(config["results_dir"]) / "map")
        self.stack = wire_stack(self.env, map_dir, RobotConfig.load())
        self.arm = self.stack.robot._arm  # chefbot ArmStack: ik, frames, arm, gripper, collision, workspace()

    # -- scene -------------------------------------------------------------------

    def apply(self, scene) -> None:
        """Robot at the fixed pose, arm folded back, gripper open; the scene's cup and
        distractors on the counter, everything else parked; then let it settle."""
        env = self.env
        env.reset_scene()
        env.teleport_base(self.base)
        env.set_arm_joints(np.array(self.config["observe_arm"]))  # folded back: out of the shoulder camera's view
        used = {f"cup:{scene.cup_model}": (scene.cup_xy, scene.cup_yaw)}
        for d in scene.distractors:
            used[d["key"]] = (tuple(d["xy"]), float(d["yaw"]))
        for i, key in enumerate(self.objects):
            if key in used:
                xy, yaw = used[key]
                world = to_world(np.array([[xy[0], xy[1], 0.0]]), self.base)[0]
                self._put(key, [world[0], world[1], self.counter_z + self._below[key] + 0.002], yaw + self.base.yaw, float_=False)
            else:
                self._put(key, [i * 0.5, 0.0, PARK_Z], 0.0, float_=True)
        for _ in range(20):
            env.wait_for_next_tick(0.05)

    def _put(self, key: str, xyz, yaw: float, float_: bool) -> None:
        env, body = self.env, self.objects[key]
        m = env.model
        joint = int(m.body_jntadr[body])
        adr, dof = int(m.jnt_qposadr[joint]), int(m.jnt_dofadr[joint])
        with env.lock:
            env.data.qpos[adr : adr + 3] = xyz
            env.data.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            env.data.qvel[dof : dof + 6] = 0.0
            m.body_gravcomp[m.body_rootid == body] = 1.0 if float_ else 0.0
            mujoco.mj_forward(m, env.data)

    @contextlib.contextmanager
    def isolated(self, keep_root: int):
        """Every other spawned object parked while inside, then put back exactly."""
        env, m = self.env, self.env.model
        saved = {}
        with env.lock:
            for i, (key, body) in enumerate(self.objects.items()):
                if body == keep_root:
                    continue
                adr = int(m.jnt_qposadr[m.body_jntadr[body]])
                saved[adr] = env.data.qpos[adr : adr + 7].copy()
                env.data.qpos[adr : adr + 3] = [i * 0.5, 0.0, PARK_Z]
            mujoco.mj_forward(m, env.data)
        try:
            yield
        finally:
            with env.lock:
                for adr, q in saved.items():
                    env.data.qpos[adr : adr + 7] = q
                mujoco.mj_forward(m, env.data)

    def cup_root(self, scene) -> int:
        return self.objects[f"cup:{scene.cup_model}"]

    def cup_pose_robot(self, scene) -> np.ndarray:
        """The cup body's pose (4x4) in the robot frame."""
        body = self.cup_root(scene)
        T = np.eye(4)
        T[:3, :3] = self.env.data.xmat[body].reshape(3, 3)
        T[:3, 3] = self.env.data.xpos[body]
        from experiments.ortho_grasp.frames import pose_to_robot

        return pose_to_robot(T, self.base)

    # -- rendering: robosuite's own offscreen context, on the env's render thread ----
    # (a second mujoco.Renderer context in this process gets the textures wrong)

    def render(self, camera, width: int, height: int, segmentation: bool = False) -> np.ndarray:
        """RGB (H, W, 3) or segmentation (H, W, 2: id, type) for a camera name or a free MjvCamera."""
        env = self.env
        return env._render_thread.submit(self._render, camera, width, height, segmentation).result()

    def _render(self, camera, width, height, segmentation):
        sim = self.env.suite.sim
        with self.env.lock:
            if isinstance(camera, str):
                out = sim.render(camera_name=camera, width=width, height=height, segmentation=segmentation)
            else:
                ctx = sim._render_context_offscreen
                cam = ctx.cam
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.lookat[:] = camera.lookat
                cam.distance, cam.azimuth, cam.elevation = camera.distance, camera.azimuth, camera.elevation
                cam.orthographic = camera.orthographic
                out = sim.render(width=width, height=height, segmentation=segmentation)
        out = np.ascontiguousarray(out[::-1])  # MuJoCo reads bottom-up
        if segmentation:
            out = out[..., ::-1]  # robosuite gives (type, id); keep (id, type) like mujoco.Renderer
        return out

    def segment_roots(self, seg: np.ndarray) -> np.ndarray:
        """Root body id per pixel (-1 where no geom) of a segmentation render."""
        m = self.env.model
        ids, types = seg[..., 0], seg[..., 1]
        out = np.full(ids.shape, -1, dtype=int)
        geom = (types == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
        out[geom] = m.body_rootid[m.geom_bodyid[ids[geom]]]
        return out

    def close(self) -> None:
        self.env.close()


def robot_points(world: World, points_world: np.ndarray) -> np.ndarray:
    return to_robot(points_world, world.base)
