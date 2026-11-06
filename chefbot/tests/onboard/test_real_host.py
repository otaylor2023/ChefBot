"""The real-robot episode host on a fake robot: moves get there, unsafe ones stop it, a dry run sends nothing, the
base holds."""

import time

import numpy as np
import pytest

from chefbot import embodiments
from chefbot.environments.real.host import RealHost, SafetyStop
from chefbot.interface.types import EnvState, Pose2D

READY = np.array([0.0, -0.75, 0.0, -2.16, 0.0, 1.41, 0.785])


class FakeRobot:
    """An Env whose arm follows its joint targets at once, on the wall clock."""

    def __init__(self):
        self.t0 = time.monotonic()
        self.q, self.width, self.closed = READY.copy(), 0.08, False
        self.sent = {"arm": 0, "base": 0, "gripper": 0, "moves": 0}  # gripper: open/closed calls; moves: move/grasp
        self.camera_mounts = {}

    def get_state(self) -> EnvState:
        return EnvState(Pose2D(0.0, 0.0, 0.0), np.zeros(3), False, time.monotonic() - self.t0, self.q.copy(), np.zeros(7),
                        self.width, self.closed)  # fmt: skip

    def set_arm_joint_targets(self, q):
        self.sent["arm"] += 1
        self.q = np.asarray(q, dtype=float).copy()

    def set_base_target(self, pose, max_vel):
        self.sent["base"] += 1
        return True

    def set_base_velocity(self, *a, **k):
        self.sent["base"] += 1

    def set_gripper(self, closed):
        self.sent["gripper"] += 1
        self.closed = float(closed) >= 0.5

    def gripper_move(self, width, speed):
        self.sent["moves"] += 1

    def gripper_grasp(self, speed, force):
        self.sent["moves"] += 1

    def wait_for_next_tick(self, dt):
        time.sleep(dt)

    def close(self):
        pass


@pytest.fixture
def robot():
    return embodiments.load("tidybot_fr3")


def host_on(robot, tmp_path, **kw):
    fake = FakeRobot()
    h = RealHost(robot, robot.speed("fast"), tmp_path, confirm=False, env=fake, log=lambda *a: None, **kw)
    h.start_here()
    return h, fake


def arm_row(F, dz=0.0, dx=0.0, gripper=0.0):
    from chefbot.robot.control.wholebody import quat_wxyz

    return [F[0, 3] + dx, F[1, 3], F[2, 3] + dz, *quat_wxyz(F[:3, :3]), gripper]


def test_an_arm_move_gets_there_and_the_base_holds(robot, tmp_path):
    h, fake = host_on(robot, tmp_path)
    try:
        F = robot.kin.flange_pose(READY)
        row = arm_row(F, dz=-0.05)
        for _ in range(40):
            rec = h.step(row, "arm")
        assert rec["pos_error_m"] < 0.005
        assert fake.sent["arm"] > 0 and fake.sent["base"] == 0
    finally:
        h.close()


def test_a_move_toward_the_floor_never_gets_near_it(robot, tmp_path):
    # The IK (the real joint limits) stops the arm short, or the host's guard does: either way not near the floor.
    h, fake = host_on(robot, tmp_path)
    try:
        F = robot.kin.flange_pose(READY)
        try:
            for _ in range(400):
                h.step(arm_row(F, dz=-0.9), "arm")
        except RuntimeError as e:
            assert "halted" in str(e) and isinstance(h.fault, SafetyStop)
        from chefbot.interface.workspace import arm_points

        assert arm_points(fake.q)[2:, 2].min() > -0.30
    finally:
        h.close()


def test_a_table_at_the_plates_height_beyond_the_base_is_allowed(robot, tmp_path):
    from chefbot.environments.real.host import GuardedEnv

    g = GuardedEnv(FakeRobot(), robot)
    from chefbot.onboard.calibration.wrist import ik

    T = robot.kin.flange_pose(READY)
    T[:3, 3] = [0.5, 0.0, 0.16]  # fingertips about 3 cm above the plate's height, half a meter out
    q = ik(READY, T)
    assert q is not None
    g.check(q)  # no SafetyStop: it is beyond the base
    T[:3, 3] = [0.25, 0.0, 0.16]  # the same height over the base: stopped
    q = ik(READY, T)
    if q is not None:
        with pytest.raises(SafetyStop):
            g.check(q)


def test_a_dry_run_sends_nothing_but_shows_the_move(robot, tmp_path):
    h, fake = host_on(robot, tmp_path, dry_run=True)
    try:
        F = robot.kin.flange_pose(READY)
        for _ in range(40):
            rec = h.step(arm_row(F, dz=-0.05, gripper=1.0), "arm")
        assert rec["pos_error_m"] < 0.005  # the skill sees its move made
        assert fake.sent == {"arm": 0, "base": 0, "gripper": 0, "moves": 0}
        assert np.allclose(fake.q, READY)  # and the arm didn't move
    finally:
        h.close()


def test_the_bringup_skill_runs_through_the_episode_on_it(robot, tmp_path):
    from chefbot import skills
    from chefbot.runtime import models as M
    from chefbot.runtime.episode import run_episode

    fake = FakeRobot()
    host = RealHost(robot, robot.speed("fast"), tmp_path, confirm=False, env=fake, log=lambda *a: None)
    skill = skills.load("bringup/v1")
    try:
        result = run_episode(host, None, 0, skill.dir, tmp_path / "ep", M.Ledger(tmp_path / "ledger.jsonl", 0.0), None,
                             {**skill.access(), "confirm_moves": False}, record=False)  # fmt: skip
    finally:
        host.close()
    assert result["returned"]["ok"], (result, (tmp_path / "ep" / "skill_log.txt").read_text())
    assert fake.sent["arm"] > 0 and fake.sent["base"] == 0 and fake.sent["moves"] == 3


def test_it_goes_home_first_and_a_dry_run_doesnt(robot, tmp_path):
    from chefbot.environments.real.host import HOME

    fake = FakeRobot()
    fake.q = np.array(HOME) + np.radians([10, -8, 5, 6, -5, 4, 20])
    h = RealHost(robot, robot.speed("fast"), tmp_path, confirm=False, env=fake, log=lambda *a: None)
    h.go_home()
    assert np.abs(fake.q - np.array(HOME)).max() < 1e-6
    fake.q = np.array(HOME) + np.radians(15)
    q_before = fake.q.copy()
    RealHost(robot, robot.speed("fast"), tmp_path, confirm=False, env=fake, log=lambda *a: None, dry_run=True).go_home()
    assert np.allclose(fake.q, q_before)


def test_a_joint_move_gets_there(robot, tmp_path):
    from chefbot.environments.real.host import HOME

    h, fake = host_on(robot, tmp_path)
    try:
        target = np.array(HOME) + np.radians([8, -5, 4, 6, -4, 5, 15])
        for _ in range(60):
            rec = h.step([*target.tolist(), 0.0], "joints")
        assert rec["pos_error_m"] < 1e-3  # rad, the largest joint's
        assert np.abs(fake.q - target).max() < 1e-3 and fake.sent["base"] == 0
    finally:
        h.close()


def test_a_dead_arm_driver_stops_it(robot, tmp_path):
    fake = FakeRobot()
    fake.raw_state = lambda: {"arm_frozen_s": 3.0}  # the driver gone: its last joints frozen in Redis
    h = RealHost(robot, robot.speed("fast"), tmp_path, confirm=False, env=fake, log=lambda *a: None)
    h.start_here()
    try:
        with pytest.raises(RuntimeError, match="halted"):
            for _ in range(20):
                h.step(h.hold_row(), "whole")
        assert "isn't publishing" in str(h.fault)
    finally:
        h.close()


def test_a_late_state_holds_and_a_lost_one_stops_it(robot, tmp_path):
    fake = FakeRobot()
    age = {"s": 0.0}
    fake.state_age = lambda: age["s"]
    h = RealHost(robot, robot.speed("fast"), tmp_path, confirm=False, env=fake, log=lambda *a: None)
    h.start_here()
    try:
        age["s"] = 1.0  # a frame on the link
        time.sleep(0.15)  # the step under way finishes
        n = fake.sent["arm"]
        time.sleep(0.3)
        assert fake.sent["arm"] == n and h.fault is None  # held, not stopped
        age["s"] = 0.0
        h.step(h.hold_row(), "whole")  # and goes on
        age["s"] = 5.0
        time.sleep(0.3)
        assert h.fault is not None and "lost" in str(h.fault)
    finally:
        h.close()


def test_a_low_tilted_grasp_past_an_elbow_limit_is_reached(robot, tmp_path):
    # 2026-11-03, chefbot_v30_fr3 on a spice jar: the old per-tick IK pinned joint 4 at its limit and stalled 0.35 m
    # short ("grasp pose not reached"), though the arm reaches it (shoulder forward, wrist bent: joint 6 ~235 deg).
    h, fake = host_on(robot, tmp_path)
    try:
        row = [0.3881, 0.0303, 0.1511, -0.1564, 0.87, -0.3109, 0.3493, 0.0]
        for _ in range(250):  # fast profile: well under 250 control steps
            rec = h.step(row, "arm")
            if rec["pos_error_m"] < 0.003:
                break
        assert rec["pos_error_m"] < 0.003 and rec["rot_error_rad"] < 0.02, rec
        lim = np.asarray(robot.kin.ik.joint_limits)
        assert np.all(fake.q >= lim[:, 0]) and np.all(fake.q <= lim[:, 1])
        assert abs(fake.q[0]) < np.radians(20)  # no swing round: joint 1 stays near forward
    finally:
        h.close()
