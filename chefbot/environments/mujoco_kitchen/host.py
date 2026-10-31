"""An episode host on a MuJoCo kitchen (env.py): the RoboCasa host's surface (environments/robocasa/host.py) on plain
MuJoCo, with the TidyBot++ FR3 and any generated extras the setup names (a pair of tongs).

    setup.scene: {layout, style, objects: [{category, model}], extras: {tongs: {generator, seed}}}

The kitchen is exported from RoboCasa once and cached (export.py); generated extras are made once per (generator,
seed) and cached beside the kitchens. Video: the robot's two cameras over two views from beside the robot.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from chefbot.embodiments import Robot, Speed
from chefbot.embodiments.wholebody import WholeBodyController, quat_wxyz
from chefbot.environments.mujoco_kitchen.env import MujocoKitchenEnv
from chefbot.environments.mujoco_kitchen.export import ensure_kitchen, kitchens_root
from chefbot.environments.mujoco_kitchen.scene import Scene

CONTROL_HZ = 15.0
TICK_S = 0.05
SETTLE_TICKS = 10
GENERATORS = Path(__file__).resolve().parents[2] / "generators"


def generated(generator: str, seed: int) -> Path:
    """A generated asset's model.xml (generators/<family>/<name>, made once per seed, cached)."""
    out = kitchens_root() / "generated" / generator.replace("/", "_") / str(seed)
    if not (out / "model.xml").exists():
        import subprocess
        import sys

        src = GENERATORS / generator / "generate.py"
        out.mkdir(parents=True, exist_ok=True)
        code = ("import importlib.util, json, sys; s = importlib.util.spec_from_file_location('g', sys.argv[1]); g = importlib.util.module_from_spec(s); "
                "s.loader.exec_module(g); json.dump(g.generate(int(sys.argv[2]), sys.argv[3]), open(sys.argv[3] + '/params.json', 'w'))")  # fmt: skip
        subprocess.run([os.environ.get("GEN_PYTHON", sys.executable), "-c", code, str(src), str(seed), str(out)], check=True, cwd=out)
    return out / "model.xml"


class MujocoKitchenHost:
    realtime = False

    def __init__(self, setup, robot: Robot, speed: Speed, out_dir: Path, record: bool = True, camera_noise: bool = True):
        self.setup, self.robot, self.out, self.record = setup, robot, Path(out_dir), record
        sc = setup.scene
        layout, style = int(sc["layout"]), int(sc["style"])
        kitchen = ensure_kitchen(layout, style, [dict(o) for o in sc["objects"]])
        extras = {name: generated(e["generator"], int(e["seed"])) for name, e in (sc.get("extras") or {}).items()}
        self.extra_params = {n: json.loads((p.parent / "params.json").read_text()) for n, p in extras.items()}
        self.env = MujocoKitchenEnv(kitchen, extras=extras)
        self.scene = Scene(self.env)
        self.noise_key = [layout, style, 7]
        self.ctl = WholeBodyController(self.env, robot, speed, tick_s=TICK_S, on_tick=self._frame)
        self.cameras = robot.cameras
        self.noise = {c["sim"]: c["noise"] for c in self.cameras.values() if camera_noise and c.get("noise")}
        self.frames, self.step_count, self.frame_every, self._ticks = [], 0, 2, 0
        self.task = None

    # -- episode -------------------------------------------------------------------

    def start(self, task, seed: int):
        self.task, self.seed = task, seed
        for pose in task.type.start_poses(self.scene, task, self.robot, seed):
            self._reset_at(pose)
            if not self.touching() and task.type.clear(self.scene, task):
                self.start_pose = pose
                task.type.started(self.scene, task, seed) if hasattr(task.type, "started") else None
                return pose
        raise RuntimeError(f"no clear start pose for {self.setup.name} / {task.name} at seed {seed}")

    def start_here(self):
        self.ctl.reset()
        self.frames, self.step_count, self._ticks = [], 0, 0
        self.out.mkdir(parents=True, exist_ok=True)
        return self.env.get_state().base_pose

    def _reset_at(self, pose) -> None:
        env = self.env
        env.reset_scene()
        env.physics_hooks.clear()
        env.teleport_base(pose)
        env.set_arm_joints(np.array(self.robot.raw.get("start_q") or self.robot.config.arm_ready))
        env.set_gripper(0.0)
        self.task.type.reset_at(self.scene, self.task, pose, self.seed)
        self._jitter_cameras(self.seed)
        mujoco.mj_forward(env.model, env.data)
        for _ in range(SETTLE_TICKS):
            env.wait_for_next_tick(TICK_S)
        self.gap_open = max(env.gripper_width(), 1e-3)
        self.ctl.reset()
        self.frames, self.step_count, self._ticks = [], 0, 0
        self.out.mkdir(parents=True, exist_ok=True)

    def _jitter_cameras(self, seed: int) -> None:
        m = self.env.model
        if not hasattr(self, "_cam_nominal"):
            self._cam_nominal = {c: (m.cam_pos[m.camera(c).id].copy(), m.cam_quat[m.camera(c).id].copy(), float(m.cam_fovy[m.camera(c).id]))
                                 for c in self.noise}  # fmt: skip
        rng = np.random.default_rng([seed, *self.noise_key])
        for c, n in self.noise.items():
            dp, dr, df = n["pos_m"], n["rot_deg"], n["fov"]
            pos, quat, fovy = self._cam_nominal[c]
            i = m.camera(c).id
            m.cam_pos[i] = pos + rng.uniform(-dp, dp, 3)
            turn = Rotation.from_rotvec(np.radians(rng.uniform(-dr, dr, 3)))
            x, y, z, w = (turn * Rotation.from_quat([*quat[1:], quat[0]])).as_quat()
            m.cam_quat[i] = [w, x, y, z]
            m.cam_fovy[i] = fovy * (1.0 + rng.uniform(-df, df))

    def step(self, row, mode: str = "whole") -> dict:
        rec = self.ctl.step(row, mode)
        self.step_count += 1
        return rec

    def close(self) -> None:
        self.env.close()

    # -- what a skill sees ---------------------------------------------------------

    def proprio(self) -> dict:
        s = self.env.get_state()
        F = self.ctl.flange_in_odom(s.arm_q)
        bx, by, byaw = self.ctl.base_in_odom()
        q_cmd = self.ctl.q_cmd
        return {"step": self.step_count, "sim_time_s": self.step_count / CONTROL_HZ, "flange_pose_odom": F.tolist(),
                "base_pose_odom": [bx, by, byaw], "gripper_cmd": float(self.env.gripper_cmd),
                "gripper_closed_cmd": bool(s.gripper_closed_cmd),
                "gripper_closure": float(np.clip(1.0 - s.gripper_width / self.gap_open, 0.0, 1.0)),
                "gripper_width_m": float(s.gripper_width), "arm_q": [float(v) for v in s.arm_q],
                "arm_q_cmd": None if q_cmd is None else [float(v) for v in q_cmd], "base_in_contact": bool(s.base_in_contact)}  # fmt: skip

    def hold_row(self) -> list[float]:
        p = self.proprio()
        F = np.array(p["flange_pose_odom"])
        return [*F[:3, 3].tolist(), *quat_wxyz(F[:3, :3]), p["gripper_cmd"], *p["base_pose_odom"]]

    def rgbd(self) -> dict[str, dict]:
        inv = np.linalg.inv(self.ctl.odom)
        out = {}
        for name, c in self.cameras.items():
            r = self.env.get_camera_rgbd(c["sim"])
            out[name] = {"rgb": r.rgb, "depth": r.depth if c.get("depth", True) else None, "K": r.intrinsics, "D": None, "model": "pinhole",
                         "cam_to_odom": inv @ r.cam_to_world}  # fmt: skip
        return out

    # -- what only the harness sees ------------------------------------------------

    def measure(self) -> dict:
        return self.task.type.measure(self.scene, self.task)

    def touching(self) -> list[str]:
        """Bodies outside the robot that it touches (not the floor)."""
        m, env = self.env.model, self.env
        robot = env.robot_geoms
        names = set()
        for a, b in env._touching():
            if (a in robot) != (b in robot):
                o = b if a in robot else a
                if o not in env.floor_geoms:
                    names.add(m.body(int(m.body_rootid[m.geom_bodyid[o]])).name)
        return sorted(names)[:8]

    def debug(self) -> dict:
        inv = np.linalg.inv(self.ctl.odom)
        centers = {n: (inv @ np.append(self.scene.object(n).center(), 1.0))[:3].round(3).tolist() for n in self.scene.objects()}
        for n in self.env.extras:
            centers[n] = (inv @ np.append(self.scene.extra(n).center(), 1.0))[:3].round(3).tolist()
        return {"objects_odom": centers, "touching": self.touching()}

    # -- recording -----------------------------------------------------------------

    def _views(self) -> list:
        """Two views from beside the robot, looking at what is in front of it."""
        s = self.env.get_state().base_pose
        cams = []
        for side in (1.0, -1.0):
            c = mujoco.MjvCamera()
            fx, fy = math.cos(s.yaw), math.sin(s.yaw)
            c.lookat[:] = [s.x + 0.55 * fx, s.y + 0.55 * fy, 0.95]
            c.distance, c.elevation = 1.1, -28.0
            c.azimuth = math.degrees(s.yaw) + 180.0 - side * 55.0
            cams.append(c)
        return cams

    def _frame(self) -> None:
        if not self.record:
            return
        self._ticks += 1
        if self._ticks % self.frame_every:
            return
        views = [self.env.render(c["sim"], 320, 240) for c in list(self.cameras.values())[:2]]
        views += [self.env.render(c, 320, 240) for c in self._views()]
        self.frames.append(np.concatenate([np.concatenate(views[:2], 1), np.concatenate(views[2:4], 1)], 0))

    def save_video(self, name: str = "video.mp4") -> Path | None:
        import imageio.v3 as iio

        if not self.frames:
            return None
        p = self.out / name
        iio.imwrite(p, np.stack(self.frames), fps=10, codec="libx264")
        return p
