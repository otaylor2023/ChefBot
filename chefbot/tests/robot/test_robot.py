import math
import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from chefbot.interface.types import MotionStatus, Pose2D, wrap_angle
from chefbot.robot.config import RobotConfig
from chefbot.robot.nav.grid import OccupancyGrid
from chefbot.robot.robot import Robot
from chefbot.tests.robot.fake_env import FakeEnv

CFG = RobotConfig(footprint_radius=0.2, safety_margin=0.05)
RES = 0.05


class RingReach:
    """Reachable when the target is 0.4-0.7 m from the base, horizontally."""

    def is_reachable(self, base_pose, target_world):
        return 0.4 <= math.hypot(target_world[0] - base_pose.x, target_world[1] - base_pose.y) <= 0.7


def room(size=6.0, blocks=()):
    """A walled square room with rectangular obstacles (x0, y0, x1, y1) in meters."""
    n = round(size / RES)
    occupied = np.zeros((n, n), dtype=bool)
    occupied[[0, -1], :] = occupied[:, [0, -1]] = True
    for x0, y0, x1, y1 in blocks:
        occupied[round(y0 / RES) : round(y1 / RES), round(x0 / RES) : round(x1 / RES)] = True
    return OccupancyGrid(occupied, RES, (0.0, 0.0))


def make(grid=None, start=Pose2D(1.0, 1.0, 0.0), cfg=CFG, **env_kwargs):
    env = FakeEnv(start, **env_kwargs)
    return Robot(env, grid or room(), cfg, RingReach()), env


def assert_stopped(env):
    assert np.all(env.commands[-1][1] == 0.0), "velocity must be zero after a motion exits"


def test_reaches_goal_within_tolerance():
    robot, env = make(room(blocks=[(2.5, 0.0, 3.0, 4.0)]))
    goal = Pose2D(5.0, 1.0, math.pi / 2)
    robot.go_to_pose(goal)
    result = robot.wait()
    assert result.status == MotionStatus.SUCCESS
    assert result.final_pose.distance_to(goal) <= CFG.pos_tolerance + 1e-3
    assert abs(wrap_angle(result.final_pose.yaw - goal.yaw)) <= CFG.yaw_tolerance + 1e-3
    assert_stopped(env)


def test_pure_rotation():
    robot, env = make()
    robot.go_to_pose(Pose2D(1.0, 1.0, -2.5))
    result = robot.wait()
    assert result.status == MotionStatus.SUCCESS
    assert abs(wrap_angle(result.final_pose.yaw + 2.5)) <= CFG.yaw_tolerance + 1e-3


def test_respects_velocity_and_acceleration_limits():
    robot, env = make(room(blocks=[(2.5, 0.0, 3.0, 4.0)]))
    robot.go_to_pose(Pose2D(5.0, 1.0, math.pi))
    assert robot.wait().status == MotionStatus.SUCCESS
    # Drop the zeroing command issued on exit: stopping is not acceleration-limited.
    commands = np.array([c for _, c in env.commands[:-1]])
    lin_speed = np.linalg.norm(commands[:, :2], axis=1)
    assert lin_speed.max() <= CFG.max_lin_vel + 1e-9
    assert lin_speed.max() > 0.9 * CFG.max_lin_vel  # it does get up to speed
    assert np.abs(commands[:, 2]).max() <= CFG.max_ang_vel + 1e-9
    dt = CFG.control_dt
    steps = np.diff(np.vstack([np.zeros(3), commands]), axis=0)
    assert np.linalg.norm(steps[:, :2], axis=1).max() <= CFG.max_lin_acc * dt + 1e-9
    assert np.abs(steps[:, 2]).max() <= CFG.max_ang_acc * dt + 1e-9


def test_blocked_on_contact():
    # A physical obstacle the map does not know about.
    hidden = room(blocks=[(2.5, 0.0, 3.0, 6.0)])
    robot, env = make(room(), obstacles=hidden)
    robot.go_to_pose(Pose2D(5.0, 1.0, 0.0))
    result = robot.wait()
    assert result.status == MotionStatus.BLOCKED
    assert result.final_pose.x < 2.5
    assert_stopped(env)


def test_can_drive_away_after_being_blocked():
    hidden = room(blocks=[(2.5, 0.0, 3.0, 6.0)])
    robot, env = make(room(), obstacles=hidden)
    robot.go_to_pose(Pose2D(5.0, 1.0, 0.0))
    assert robot.wait().status == MotionStatus.BLOCKED
    robot.go_to_pose(Pose2D(1.0, 1.0, 0.0))
    assert robot.wait().status == MotionStatus.SUCCESS


def test_timeout():
    robot, env = make(cfg=replace(CFG, motion_timeout_s=1.0))
    robot.go_to_pose(Pose2D(5.0, 5.0, 0.0))
    result = robot.wait()
    assert result.status == MotionStatus.TIMEOUT
    assert 1.0 <= env.time <= 1.0 + 3 * CFG.control_dt
    assert_stopped(env)


def test_stop_base_stops_within_one_control_tick():
    robot, env = make(tick_sleep=0.002)
    robot.go_to_pose(Pose2D(5.0, 5.0, 0.0))
    time.sleep(0.1)
    assert robot.get_status().status == MotionStatus.MOVING
    ticks_before = env.ticks
    robot.stop_base()
    # At most the tick already in flight, plus the settling tick after the stop.
    assert env.ticks - ticks_before <= 2
    result = robot.get_status()
    assert result.status == MotionStatus.STOPPED
    assert_stopped(env)
    pose = env.pose
    time.sleep(0.05)
    assert env.pose == pose and env.ticks - ticks_before <= 2


def test_stop_base_when_idle_is_harmless():
    robot, env = make()
    robot.stop_base()
    assert robot.get_status().status == MotionStatus.IDLE
    assert robot.wait().status == MotionStatus.IDLE
    assert_stopped(env)


def test_new_motion_preempts_old():
    robot, env = make(tick_sleep=0.002)
    robot.go_to_pose(Pose2D(5.0, 5.0, 0.0))
    first = {}
    waiter = threading.Thread(target=lambda: first.update(result=robot.wait()))
    waiter.start()
    time.sleep(0.05)
    env.tick_sleep = 0.0
    second_goal = Pose2D(1.5, 1.0, 0.0)
    robot.go_to_pose(second_goal)
    waiter.join(timeout=5.0)
    assert first["result"].status == MotionStatus.STOPPED
    result = robot.wait()
    assert result.status == MotionStatus.SUCCESS
    assert result.final_pose.distance_to(second_goal) <= CFG.pos_tolerance + 1e-3


def test_wait_returns_on_completion_and_moving_on_timeout():
    robot, env = make(tick_sleep=0.002)
    robot.go_to_pose(Pose2D(3.0, 1.0, 0.0))
    early = robot.wait(timeout_s=0.05)
    assert early.status == MotionStatus.MOVING
    assert early.final_pose is not None
    env.tick_sleep = 0.0
    assert robot.wait().status == MotionStatus.SUCCESS  # the motion kept going
    assert robot.get_status().status == MotionStatus.SUCCESS


def test_goal_in_collision_snaps_to_nearest_free_cell():
    grid = room(blocks=[(3.0, 3.0, 4.0, 4.0)])
    robot, env = make(grid)
    goal = Pose2D(3.1, 3.5, 1.0)  # just inside the block's left face
    robot.go_to_pose(goal)
    result = robot.wait()
    assert result.status == MotionStatus.SUCCESS
    inflated = grid.inflate(CFG.inflation_radius)
    assert inflated.is_free(result.final_pose.x, result.final_pose.y)
    assert result.final_pose.x < 3.0 and abs(result.final_pose.y - 3.5) < 0.1
    assert result.final_pose.distance_to(goal) <= CFG.snap_radius
    assert abs(wrap_angle(result.final_pose.yaw - 1.0)) <= CFG.yaw_tolerance + 1e-3


def test_goal_too_deep_in_collision_is_no_path():
    robot, env = make(room(blocks=[(2.0, 2.0, 5.0, 5.0)]))
    robot.go_to_pose(Pose2D(3.5, 3.5, 0.0))
    result = robot.wait()
    assert result.status == MotionStatus.NO_PATH
    assert env.ticks == 0


def test_enclosed_goal_is_no_path():
    # A free pocket walled off from the rest of the room.
    grid = room(blocks=[(3.0, 3.0, 5.0, 3.1), (3.0, 4.9, 5.0, 5.0), (3.0, 3.0, 3.1, 5.0), (4.9, 3.0, 5.0, 5.0)])
    robot, env = make(grid)
    robot.go_to_pose(Pose2D(4.0, 4.0, 0.0))
    assert robot.wait().status == MotionStatus.NO_PATH


def test_align_picks_reachable_standoff_facing_target():
    # Target on a counter along the far wall.
    grid = room(blocks=[(2.0, 4.0, 5.0, 6.0)])
    robot, env = make(grid, start=Pose2D(3.0, 2.0, 0.0))
    target = np.array([3.5, 4.2, 0.9])
    robot.align_to_point(target)
    result = robot.wait()
    assert result.status == MotionStatus.SUCCESS
    pose = result.final_pose
    assert grid.inflate(CFG.inflation_radius).is_free(pose.x, pose.y)
    assert RingReach().is_reachable(pose, target)
    bearing = math.atan2(target[1] - pose.y, target[0] - pose.x)
    assert abs(wrap_angle(bearing - pose.yaw)) <= math.radians(3.0)
    assert_stopped(env)


def test_align_unreachable_when_target_surrounded_by_obstacles():
    robot, env = make(room(blocks=[(2.0, 2.0, 5.0, 5.0)]))
    robot.align_to_point(np.array([3.5, 3.5, 0.9]))
    result = robot.wait()
    assert result.status == MotionStatus.UNREACHABLE
    assert env.ticks == 0


def test_align_unreachable_when_arm_cannot_reach():
    class NeverReach:
        def is_reachable(self, base_pose, target_world):
            return False

    env = FakeEnv(Pose2D(1.0, 1.0, 0.0))
    robot = Robot(env, room(), CFG, NeverReach())
    robot.align_to_point(np.array([3.0, 3.0, 0.9]))
    assert robot.wait().status == MotionStatus.UNREACHABLE


def test_default_reachability_uses_fr3_ik():
    env = FakeEnv(Pose2D(1.0, 1.0, 0.0))
    robot = Robot(env, room(), replace(CFG, arm_mount_xyz=(0.1, 0.0, 0.5)))
    robot.align_to_point(np.array([3.0, 3.0, 0.9]))
    result = robot.wait()
    assert result.status == MotionStatus.SUCCESS
    assert math.hypot(3.0 - result.final_pose.x, 3.0 - result.final_pose.y) <= CFG.standoff_max + 0.02
    robot.align_to_point(np.array([3.0, 3.0, 2.5]))  # far above the arm
    assert robot.wait().status == MotionStatus.UNREACHABLE


def test_motion_error_is_reported_not_raised():
    class Broken:
        def is_reachable(self, base_pose, target_world):
            raise RuntimeError("ik exploded")

    env = FakeEnv(Pose2D(1.0, 1.0, 0.0))
    robot = Robot(env, room(), CFG, Broken())
    robot.align_to_point(np.array([3.0, 3.0, 0.9]))
    result = robot.wait()
    assert result.status == MotionStatus.ERROR
    assert "ik exploded" in result.message
