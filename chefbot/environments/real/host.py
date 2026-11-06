"""The real robot as an episode host: what runtime/episode.py drives in place of a simulator, so a lab skill
(interface chefbot-v1) runs on the robot unchanged.

The same contract as the simulated hosts (environments/mujoco_kitchen/host.py), with real time: a control thread
runs the same whole-body controller as the sims (robot/control/wholebody.py: IK every tick, joint targets to the arm)
on RealEnv (the onboard server), each step toward the latest target, so the robot keeps holding its last command
while the skill thinks or waits on a model. Robot time counts from the skill's first request.

Between the controller and the robot, every arm command is checked (SafetyStop, and the robot holds, if any fails):
the joints inside their limits, no part of the arm in the keep-out of the robot's own parts (the embodiment's body:
the pole, the camera on its clamp; interface/workspace.py) or within its plate clearance of the base's plate, or the
flange below `flange_z_min` over the plate (within its outline; beyond it a table may be at the plate's height or
lower), nothing near the floor (arm base frame). The controller steers clear of the same body (embodiments/motion.py),
a few cm further out: this guard is the backstop. The base moves only if the host
is made with base=True (an arm-only skill never needs it). dry_run: nothing is sent to the arm or the gripper; the
arm's joints are taken to be where the controller commanded them (the skill sees its moves made), the cameras are
the real ones (seeing the robot where it really is). confirm: Enter before each move (episode.py, the skill's
`confirm_moves`).

Nothing measures the task on the real robot: measure() is empty and runs are unscored.

(agentic-skills-exploration's environments/real/host.py, on ChefBot's RealEnv and controller.)
"""

from __future__ import annotations

import dataclasses
import math
import threading
import time
from pathlib import Path

import numpy as np

from chefbot.embodiments import Robot, Speed
from chefbot.robot.control.wholebody import WholeBodyController, quat_wxyz
from chefbot.interface.types import Pose2D

HOME = (0.0, -0.75, 0.0, -2.16, 0.0, 1.41, 0.785)  # the FR3's ready pose: the hand pointing down (the calibrations' start)
HOME_SPEED = 0.3  # rad/s, the fastest joint, going home
# Over the base's plate (within its outline: the embodiment's body), every arm point at least its plate clearance
# above it, the flange above FLANGE_Z_MIN. Beyond the base, a table may be at the plate's height or lower: there only
# the floor (0.44 m below the plate on the real base) bounds it.
FLANGE_Z_MIN = 0.12  # m above the plate, over it
FLOOR_CLEAR_Z = -0.30  # m: every arm point above this anywhere (the floor is at -0.44 on the real base: plate 44 cm up)
OPEN_WIDTH = 0.08  # m: the Franka Hand fully open
ARM_FROZEN_S = 0.5  # the arm's joints unchanged this long (the onboard server's arm_frozen_s): its driver is gone
# The robot's state can lag while a camera frame (~2 MB) crosses the link ahead of it: the control loop holds (sends
# nothing new; the arm holds its last target) while it is STATE_LATE_S..STATE_LOST_S old, and stops only past that.
STATE_LATE_S, STATE_LOST_S = 0.4, 3.0


class SafetyStop(RuntimeError):
    """An arm command the host won't send."""


class GuardedEnv:
    """RealEnv with every arm command checked, the base locked unless allowed, and (dry_run) no actuation at all."""

    def __init__(self, env, robot: Robot, base: bool = False, dry_run: bool = False, flange_z_min: float = FLANGE_Z_MIN):
        from chefbot.interface.workspace import LIMITS, MARGIN

        from chefbot.interface.workspace import BODY

        self.env, self.kin, self.base, self.dry_run, self.flange_z_min = env, robot.kin, base, dry_run, flange_z_min
        self.body = robot.body or BODY
        from chefbot.robot.control.constraints import for_robot

        self.constraints = robot.kin.constraints or for_robot(self.body)
        self.lo, self.hi = LIMITS[:, 0] + MARGIN / 2, LIMITS[:, 1] - MARGIN / 2
        self._q = None  # dry run: the arm where it was commanded
        self._last_q = None  # the last joints sent (a step out of a keep-out the arm is in passes the guard)
        self._width = None

    def __getattr__(self, name):  # everything not overridden (cameras, halt, close, raw_state, ...) passes through
        return getattr(self.env, name)

    def get_state(self):
        s = self.env.get_state()
        if not self.dry_run:
            return s
        if self._q is None:
            self._q = np.asarray(s.arm_q, dtype=float).copy()
        return dataclasses.replace(s, arm_q=self._q.copy(), arm_qd=np.zeros_like(self._q),
                                   gripper_width=s.gripper_width if self._width is None else self._width)  # fmt: skip

    def check(self, q) -> None:
        from chefbot.interface.workspace import arm_points

        q = np.asarray(q, dtype=float)
        if np.any(q < self.lo) or np.any(q > self.hi):
            raise SafetyStop(f"joints outside their limits: {np.degrees(q).round(1).tolist()}")
        body = self.body
        why = self.violation(q)
        if why:
            c = self.clearance(q)
            ref = self._last_q if self._last_q is not None else np.asarray(self.get_state().arm_q, dtype=float)
            if c < self.clearance(ref) - 1e-5:  # in, and deeper: stop (moving out is let be)
                raise SafetyStop(f"the arm would go into {why}")
        pts = arm_points(q, body.hand_m)[2:]
        if pts[:, 2].min() < FLOOR_CLEAR_Z:
            raise SafetyStop(f"the arm would go down to {pts[:, 2].min():.2f} m, near the floor")
        F = self.kin.flange_pose(q)
        if body.over_plate(F[:3, 3])[0] and F[2, 3] < self.flange_z_min:
            raise SafetyStop(f"the flange would go to {F[2, 3]:.3f} m over the base, below its floor of {self.flange_z_min} m (arm base frame)")

    def violation(self, q) -> str | None:
        """The first of the arm's constraints (robot/control/constraints.py, the same the solvers avoid) joints q
        violate, said, or None."""
        from chefbot.robot.control.constraints import KeepOut
        from chefbot.interface.workspace import arm_points

        pts = arm_points(np.asarray(q, dtype=float), self.body.hand_m)
        for c in self.constraints:
            d, _ = c.clearance(pts)
            if (d < 0).any():
                what = f"the keep-out of the {c.nearest_part(pts)}" if isinstance(c, KeepOut) else c.name
                return f"{what} ({-d.min() * 100:.1f} cm in)"
        return None

    def clearance(self, q) -> float:
        """The arm's least clearance (m; negative: in) from its constraints at joints q."""
        from chefbot.robot.control.constraints import nearest
        from chefbot.interface.workspace import arm_points

        return float(nearest(self.constraints, arm_points(np.asarray(q, dtype=float), self.body.hand_m))[0].min())

    def path_clear(self, q0, q1, step: float = 0.02) -> bool:
        """The joint-space line q0 -> q1 clear of the constraints all the way (from a start inside one: never further
        in, as the solvers move)."""
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        k = max(1, int(np.ceil(np.abs(q1 - q0).max() / step)))
        floor = min(0.0, self.clearance(q0))
        for t in np.linspace(0, 1, k + 1)[1:]:
            c = self.clearance(q0 + (q1 - q0) * t)
            if c < floor - 1e-5:
                return False
            floor = max(floor, min(0.0, c))
        return True

    def set_arm_joint_targets(self, q) -> None:
        self.check(q)
        self._last_q = np.asarray(q, dtype=float).copy()
        if self.dry_run:
            self._q = np.asarray(q, dtype=float).copy()
            return
        self.env.set_arm_joint_targets(q)

    def set_arm_pose_target(self, flange_pose) -> bool:
        raise SafetyStop("the host commands the arm by joints only (each one checked)")

    def set_base_target(self, pose, max_vel) -> bool:
        if self.base and not self.dry_run:
            return self.env.set_base_target(pose, max_vel)
        return True  # held (the base doesn't move): the controller needn't drive it either

    def set_base_velocity(self, vx, vy, vyaw, frame: str = "world") -> None:
        if self.base and not self.dry_run:
            self.env.set_base_velocity(vx, vy, vyaw, frame)

    def set_gripper(self, closed) -> None:
        if self.dry_run:
            self._width = 0.0 if float(closed) >= 0.5 else OPEN_WIDTH
            return
        self.env.set_gripper(closed)

    def gripper_move(self, width: float, speed: float) -> None:
        if self.dry_run:
            self._width = float(width)
            return
        self.env.gripper_move(width, speed)

    def gripper_grasp(self, speed: float, force: float) -> None:
        if self.dry_run:
            self._width = 0.0
            return
        self.env.gripper_grasp(speed, force)


class RealHost:
    realtime = True

    def __init__(self, robot: Robot, speed: Speed, out_dir: Path, record: bool = False, dry_run: bool = False,
                 confirm: bool = True, base: bool = False, url: str | None = None, flange_z_min: float = FLANGE_Z_MIN,
                 log=print, env=None, cameras=None):  # fmt: skip
        """env: the robot's Env (default: RealEnv from environments/real/config.yaml; tests give a fake).
        cameras: the skill's (its manifest's), held open on the robot for the run; None: all of them."""
        self.robot, self.out, self.record, self.log, self.setup = robot, Path(out_dir), record, log, None
        self.confirm, self.dry_run, self.n_moves = confirm, dry_run, 0
        kin = robot.kin
        mount = kin.root_in_world(Pose2D(0.0, 0.0, 0.0))  # the arm base on the base
        if env is None:
            from chefbot.environments.real.env import RealEnv

            # flange_to_control serves pose targets only, which the host never sends (GuardedEnv refuses them)
            env = RealEnv.from_config(np.eye(4), url=url, flange_in_base=lambda q: mount @ kin.flange_pose(q))
        self.raw_env = env
        keep = getattr(env, "keep_cameras_open", None)
        if keep is not None:
            keep(list(env.camera_mounts) if cameras is None else list(cameras))
        self.env = GuardedEnv(self.raw_env, robot, base=base, dry_run=dry_run, flange_z_min=flange_z_min)
        self.speed = robot.on_real(speed)  # never above the robot's real-only limits, whatever the run asked
        if self.speed != speed:
            log(f"[real] speed {speed.name} clipped to the real robot's limits: base {self.speed.base_max_vel}, "
                f"arm {self.speed.arm_lin} m/s, joints {self.speed.joint} rad/s")
        self.ctl = WholeBodyController(self.env, robot, self.speed)
        self.hz = self.ctl.hz
        self.cond = threading.Condition()
        self.step_count, self.frames, self.fault = 0, [], None
        self._thread, self._stop, self._clock_on, self._row, self._rec = None, False, False, None, None
        if dry_run:
            log("[real] dry run: nothing is sent to the arm or the gripper; the cameras are real")
        if not base:
            log("[real] the base holds (arm only)")

    # -- episode -------------------------------------------------------------------

    def go_home(self, q_home=HOME, confirm: bool | None = None) -> None:
        """Before the episode: the arm to `q_home` along a straight line in joint space (its path clear of the
        keep-out, every step checked), Enter first (confirm, default the host's). A dry run doesn't move: it only says
        whether the arm is there (the cameras see the real arm)."""
        self.check_live()
        q0 = np.asarray(self.raw_env.get_state().arm_q, dtype=float)
        q1 = np.asarray(q_home, dtype=float)
        far = float(np.abs(q1 - q0).max())
        if far < 0.02:
            return
        if self.dry_run:
            self.log(f"[real] dry run: the arm is {np.degrees(far):.0f} deg from home and won't go there; the skill sees it where it is")
            return
        if not self.env.path_clear(q0, q1):
            raise SafetyStop("the straight way home passes the pole's keep-out: move the arm nearer home first")
        if self.confirm if confirm is None else confirm:
            input(f"\nHome: the arm to its ready pose, {np.degrees(far):.0f} deg at most ({far / HOME_SPEED:.1f} s). Enter to move (Ctrl-C stops): ")
        tick = 0.05
        n = max(1, int(math.ceil(far / (HOME_SPEED * tick))))
        for k in range(1, n + 1):
            self.env.set_arm_joint_targets(q0 + (q1 - q0) * (k / n))
            self.raw_env.wait_for_next_tick(tick)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 5.0 and np.abs(np.asarray(self.raw_env.get_state().arm_q) - q1).max() > 0.02:
            self.raw_env.wait_for_next_tick(tick)
        err = float(np.abs(np.asarray(self.raw_env.get_state().arm_q) - q1).max())
        self.log(f"[real] home ({np.degrees(err):.1f} deg off)")

    def start(self, task=None, seed: int = 0) -> Pose2D:
        """The robot starts where it is: odom is fixed here, and control begins."""
        return self.start_here()

    def start_here(self) -> Pose2D:
        self.ctl.reset()
        self.step_count, self.frames, self.fault, self._stop, self._clock_on, self._row = 0, [], None, False, False, None
        self._hold = self.hold_row()
        self.out.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._control, name="real-host-control", daemon=True)
        self._thread.start()
        return self.env.get_state().base_pose

    def check_live(self) -> None:
        """SafetyStop if the arm driver isn't publishing (its last joints stay in Redis, frozen: commands go nowhere)."""
        raw = getattr(self.raw_env, "raw_state", None)
        frozen = (raw() or {}).get("arm_frozen_s") if raw is not None else None
        if frozen is not None and frozen > ARM_FROZEN_S:
            raise SafetyStop(f"the arm driver isn't publishing (the joints frozen {frozen:.1f} s): is it running, the arm unlocked?")

    def _control(self) -> None:
        """The controller, step after step toward the latest target (holding where it started until there is one)."""
        late = getattr(self.raw_env, "state_age", None)
        if late is not None:
            self.raw_env.max_state_age_s = STATE_LOST_S  # get_state itself fails only past this
        while not self._stop:
            try:
                if late is not None and late() > STATE_LATE_S:
                    if late() > STATE_LOST_S:
                        raise RuntimeError(f"no state from the robot for {late():.1f} s: lost the onboard server?")
                    time.sleep(0.02)  # a frame on the link: hold, then go on
                    continue
                self.check_live()
                row, mode = self._row or (self._hold, "whole")
                rec = self.ctl.step(row, mode)
            except Exception as e:  # noqa: BLE001  (a SafetyStop, the link down, the server's halt)
                self.fault = e
                self.log(f"[real] stopped: {type(e).__name__}: {e}; holding")
                self._hold_now()
                with self.cond:
                    self.cond.notify_all()
                return
            with self.cond:
                self.step_count += self._clock_on
                self._rec = rec
                self.cond.notify_all()

    def step(self, row, mode: str = "whole") -> dict:
        """Make `row` the target and return after the next control step."""
        if self.fault is not None:
            raise RuntimeError(f"robot halted: {self.fault}")
        self._clock_on = True
        with self.cond:
            self._row = ([float(v) for v in row], mode)
            n = self.step_count
            if not self.cond.wait_for(lambda: self.step_count > n or self.fault is not None, timeout=3.0):
                raise RuntimeError("control loop stalled")
        if self.fault is not None:
            raise RuntimeError(f"robot halted: {self.fault}")
        return self._rec

    def confirm_move(self, rows, mode: str = "whole") -> None:
        """Enter before the robot follows `rows`; robot time stops while waiting (the robot holds). Ctrl-C stops."""
        if not self.confirm:
            return
        p = self._proprio()
        F = np.array(p["flange_pose_odom"]) if mode == "whole" else self.robot.kin.flange_pose(p["arm_q"])
        if mode == "joints":  # as an arm move: where each row's joints put the flange (root frame), its gripper last
            rows = [[*self.robot.kin.flange_pose(np.asarray(r[:-1]))[:3, 3], 1.0, 0.0, 0.0, 0.0, r[-1]] for r in rows]
        if mode == "base":
            far, grip = 0.0, False
        else:
            far = max(np.linalg.norm(np.asarray(r[:3]) - F[:3, 3]) for r in rows)
            grip = (rows[-1][7] >= 0.5) != bool(p["gripper_closed_cmd"])
        if far < 0.02 and not grip and mode != "base":
            return  # a hold where it already is
        end = np.asarray(rows[-1], dtype=float)
        what = {"whole": "whole body", "arm": "arm only: the base holds", "joints": "arm joints: the base holds", "base": "base only: the arm holds"}[mode]
        parts = [] if mode == "base" else [f"the flange {np.round((end[:3] - F[:3, 3]) * 100, 1).tolist()} cm"
                                           f" ({'odom' if mode == 'whole' else 'arm base frame'}), farthest {far * 100:.1f} cm,"
                                           f" gripper {'closed' if end[7] >= 0.5 else 'open'}"]  # fmt: skip
        self.n_moves += 1
        clock, self._clock_on = self._clock_on, False
        try:
            input(f"\nMove {self.n_moves} ({what}): {len(rows)} steps ({len(rows) / self.hz:.1f} s). {'; '.join(parts)}."
                  f"{' [dry run]' if self.dry_run else ''}\nEnter to move (Ctrl-C stops): ")  # fmt: skip
        finally:
            self._clock_on = clock

    def _hold_now(self) -> None:
        try:
            q = self.raw_env.get_state().arm_q
            if not self.dry_run and q is not None:
                self.raw_env.set_arm_joint_targets(np.asarray(q, dtype=float))
        except Exception as e:  # noqa: BLE001
            self.log(f"[real] hold: {type(e).__name__}: {e}")

    def halt(self) -> None:
        self._stop = True
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._hold_now()

    def close(self) -> None:
        self.halt()
        self.raw_env.close()

    # -- what the skill sees -------------------------------------------------------------

    def gripper_move(self, width: float, speed: float) -> None:
        self.env.gripper_move(width, speed)

    def gripper_grasp(self, speed: float, force: float) -> None:
        self.env.gripper_grasp(speed, force)

    def proprio(self) -> dict:
        self._clock_on = True  # the skill's first request starts robot time
        return self._proprio()

    def _proprio(self) -> dict:
        s = self.env.get_state()
        F = self.ctl.flange_in_odom(s.arm_q)
        q_cmd = self.ctl.q_cmd
        return {"step": self.step_count, "sim_time_s": self.step_count / self.hz, "flange_pose_odom": F.tolist(),
                "base_pose_odom": list(self.ctl.base_in_odom()), "gripper_closed_cmd": bool(s.gripper_closed_cmd),
                "gripper_closure": float(np.clip(1.0 - s.gripper_width / OPEN_WIDTH, 0.0, 1.0)), "gripper_width_m": float(s.gripper_width),
                "arm_q": [float(v) for v in s.arm_q], "arm_q_cmd": None if q_cmd is None else [float(v) for v in q_cmd],
                "base_in_contact": False}  # fmt: skip

    def hold_row(self) -> list[float]:
        p = self._proprio()
        F = np.array(p["flange_pose_odom"])
        return [*F[:3, 3].tolist(), *quat_wxyz(F[:3, :3]), 1.0 if p["gripper_closed_cmd"] else 0.0, *p["base_pose_odom"]]

    def rgbd(self, names=None) -> dict[str, dict]:
        """The cameras' frames (`names`, or all), each one's pose in odom from its calibration (the pole's tracked,
        the wrist's through the arm's measured joints: where the camera really was, also in a dry run)."""
        self._clock_on = True
        inv = np.linalg.inv(self.ctl.odom)
        out = {}
        for name in self.raw_env.camera_mounts:
            if names is not None and name not in names:
                continue
            r = self.raw_env.get_camera_rgbd(name)
            out[name] = {"rgb": r.rgb, "depth": r.depth, "K": np.asarray(r.intrinsics), "D": None, "model": "pinhole",
                         "cam_to_odom": inv @ r.cam_to_world}  # fmt: skip
        return out

    # -- what only the harness sees ------------------------------------------------

    def measure(self) -> dict:
        return {}  # nothing measures the task on the real robot

    def debug(self) -> dict:
        return {}

    def save_video(self, name: str = "video.mp4") -> Path | None:
        return None  # not recorded (the cameras' frames are the skill's; images it saves are in images/)
