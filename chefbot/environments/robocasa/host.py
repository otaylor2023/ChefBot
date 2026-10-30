"""An episode host: one RoboCasa kitchen (ChefBot's RoboCasaEnv) with one lab robot, for the episode runner.

The host starts a task at a seed (the task's reset and start rule, the robot's camera noise), runs the whole-body
controller one 15 Hz step at a time, and serves what a skill may see (proprio, calibrated camera images) and what
only the harness may (the task's measurements, contacts). Time is simulated: it only passes while the host steps.

    host.start(task, seed) -> start pose      host.step(row, mode) -> tracking errors
    host.proprio(), host.rgbd(), host.hold_row()        what the skill sees
    host.measure(), host.debug()                         privileged: the harness's only
    host.frames, host.save_video(), host.step_count, host.realtime, host.close()

Cameras: each camera with `noise` is moved and turned a little and its field of view changed per (seed, kitchen);
the skill gets the true pose and intrinsics, as from a calibrated robot.
Video: the robot's two cameras beside RoboCasa's two shoulder views, at 10 fps.

(agentic-skills-exploration's environments/sim_host.py and robocasa/host.py, on ChefBot's env.)
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from chefbot.embodiments import Robot, Speed
from chefbot.environments.robocasa.scene import Scene
from chefbot.embodiments.wholebody import WholeBodyController, quat_wxyz
from chefbot.environments.robocasa.config import RoboCasaConfig
from chefbot.environments.robocasa.env import RoboCasaEnv

CONTROL_HZ = 15.0
SETTLE_TICKS = 10
VIDEO_VIEWS = ("robot0_agentview_left", "robot0_agentview_right")  # beside the robot's own cameras


def make_host(setup, robot: Robot, speed: Speed, out_dir: Path, record: bool = True, camera_noise: bool = True) -> "RoboCasaHost":
    if setup.environment != "robocasa":
        raise ValueError(f"setup {setup.name}: unknown environment {setup.environment!r}")
    return RoboCasaHost(setup, robot, speed, out_dir, record, camera_noise)


class RoboCasaHost:
    realtime = False

    def __init__(self, setup, robot: Robot, speed: Speed, out_dir: Path, record: bool = True, camera_noise: bool = True):
        self.setup, self.robot, self.out, self.record = setup, robot, Path(out_dir), record
        layout, style = int(setup.scene["layout"]), int(setup.scene["style"])
        objects = [dict(o) for o in setup.scene["objects"]]
        config = RoboCasaConfig.load(layout=layout, style=style, seed=int(setup.scene.get("seed", 0)), robot=robot.env["robot"],
                                     gripper=robot.env["gripper"], objects=tuple({**o, "surface": "counter"} for o in objects))  # fmt: skip
        self.env = RoboCasaEnv(config, render=True)
        self.env.restrict_arm_joint_limits(robot.kin.ik.joint_limits)  # the embodiment's arm (the FR3's limits on the sim Panda)
        self.scene = Scene(self.env, [o["category"] for o in objects])
        self.noise_key = [layout, style, 7]
        self.ctl = WholeBodyController(self.env, robot, speed, tick_s=1.0 / config.control_freq, on_tick=self._frame)
        self.cameras = robot.cameras  # skill name -> {sim, depth, noise}
        self.noise = {c["sim"]: c["noise"] for c in self.cameras.values() if camera_noise and c.get("noise")}
        self.frames, self.step_count, self.frame_every, self._ticks = [], 0, 2, 0  # video at 10 fps
        self.task = None

    # -- episode -------------------------------------------------------------------

    def start(self, task, seed: int):
        """Reset the world for `task` at `seed` and place the robot at the first clear start pose; returns it."""
        self.task, self.seed = task, seed
        for pose in task.type.start_poses(self.scene, task, self.robot, seed):
            self._reset_at(pose)
            if not self.touching() and task.type.clear(self.scene, task):
                self.start_pose = pose
                return pose
        raise RuntimeError(f"no clear start pose for {self.setup.name} / {task.name} at seed {seed}")

    def start_here(self):
        """Go on from the world as it is (an agent's next skill): odom fixed where the robot is now, nothing reset."""
        self.ctl.reset()
        self.frames, self.step_count, self._ticks = [], 0, 0
        self.out.mkdir(parents=True, exist_ok=True)
        return self.env.get_state().base_pose

    def _reset_at(self, pose) -> None:
        env = self.env
        env.reset_scene()
        env.teleport_base(pose)
        env.set_arm_joints(np.array(self.robot.config.arm_ready))
        env.set_gripper(False)
        with env.lock:
            self.task.type.reset_at(self.scene, self.task, pose, self.seed)
            self._jitter_cameras(self.seed)
            mujoco.mj_forward(env.model, env.data)
        for _ in range(SETTLE_TICKS):
            env.set_base_velocity(0.0, 0.0, 0.0)
            env.wait_for_next_tick(self.ctl.tick_s)
        self.gap_open = max(env.get_state().gripper_width, 1e-3)
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
                "base_pose_odom": [bx, by, byaw], "gripper_closed_cmd": bool(s.gripper_closed_cmd),
                "gripper_closure": float(np.clip(1.0 - s.gripper_width / self.gap_open, 0.0, 1.0)),
                "gripper_width_m": float(s.gripper_width), "arm_q": [float(v) for v in s.arm_q],
                "arm_q_cmd": None if q_cmd is None else [float(v) for v in q_cmd], "base_in_contact": bool(s.base_in_contact)}  # fmt: skip

    def hold_row(self) -> list[float]:
        """The current pose as a target: what the robot holds before the skill's first command."""
        p = self.proprio()
        F = np.array(p["flange_pose_odom"])
        return [*F[:3, 3].tolist(), *quat_wxyz(F[:3, :3]), 1.0 if p["gripper_closed_cmd"] else 0.0, *p["base_pose_odom"]]

    def rgbd(self) -> dict[str, dict]:
        """Each camera: rgb, depth (None without depth), K, D, model, cam_to_odom (OpenCV axes)."""
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
        touching = sorted({m.body(int(m.geom_bodyid[o])).name for a, b in env._touching() if (a in robot) != (b in robot)
                           for o in ((b,) if a in robot else (a,)) if o not in env._floor_geoms})  # fmt: skip
        return touching[:8]

    def debug(self) -> dict:
        """Privileged extras for the researcher's trace: the objects' centers in odom, and contacts."""
        inv = np.linalg.inv(self.ctl.odom)
        centers = {}
        for n in self.scene.objects():
            lo, hi = self.scene.object(n).box()
            centers[n] = (inv @ np.append((lo + hi) / 2, 1.0))[:3].round(3).tolist()
        return {"objects_odom": centers, "touching": self.touching()}

    # -- recording -----------------------------------------------------------------

    def _frame(self) -> None:
        if not self.record:
            return
        self._ticks += 1
        if self._ticks % self.frame_every:
            return
        cams = [c["sim"] for c in self.cameras.values()][:2] + list(VIDEO_VIEWS)
        views = [self.env.render(cam, 320, 240) for cam in cams[:4]]
        while len(views) < 4:
            views.append(np.zeros_like(views[0]))
        self.frames.append(np.concatenate([np.concatenate(views[:2], 1), np.concatenate(views[2:4], 1)], 0))

    def save_video(self, name: str = "video.mp4") -> Path | None:
        import imageio.v3 as iio

        if not self.frames:
            return None
        p = self.out / name
        iio.imwrite(p, np.stack(self.frames), fps=10, codec="libx264")
        return p
