"""Arm control, IK and the manipulation skills on FakeEnv (MANIP_SPEC §11, robot layer)."""

import math
import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from chefbot.interface.fixtures import CLOSED, OPEN, UNKNOWN, FixtureActuator
from chefbot.interface.proposers import GraspProposer, PlacementProposer
from chefbot.interface.types import RGBD, ActuationResult, Detection, Grasp, MotionStatus, Placement, Pose2D
from chefbot.robot.arm.arm_controller import ArmController
from chefbot.robot.arm.ik import arm_ik_from_config
from chefbot.robot.config import RobotConfig
from chefbot.robot.kinematics.reachability import ReachabilityChecker
from chefbot.robot.nav.grid import OccupancyGrid
from chefbot.robot.robot import Robot
from chefbot.tests.robot.fake_env import GRIPPER_OPEN, FakeEnv, FakeObject

CFG = RobotConfig(footprint_radius=0.2, safety_margin=0.05, arm_mount_xyz=(0.2, 0.0, 0.7))
IK = arm_ik_from_config(CFG)
FRAMES = ReachabilityChecker(CFG, IK)
STOW, CARRY, READY = (np.array(q) for q in (CFG.arm_stow, CFG.arm_carry, CFG.arm_ready))
START = Pose2D(3.0, 3.0, 0.0)
OBJECT_AT = np.array([3.75, 3.0, 0.95])  # 0.55 m in front of the arm base, at counter height
DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


def pose_at(position, rotation=DOWN) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, position
    return pose


def tcp_world(base_pose, arm_q) -> np.ndarray:
    return FRAMES.pose_to_world(base_pose, IK.fk(arm_q))[:3, 3]


class OneGrasp(GraspProposer):
    """Top-down grasp on whatever position the detection carries."""

    def __init__(self):
        self.calls = 0

    def propose(self, target):
        self.calls += 1
        if target.label == "counter":
            return []
        return [Grasp(pose_at(target.position_world), 0.10, 0.05, 1.0)]


class OnePlacement(PlacementProposer):
    def propose(self, surface, held):
        return [] if surface.label == "sink" else [Placement(pose_at(surface.position_world), 0.10, 1.0)]


class Door(FixtureActuator):
    """A door whose arc covers the floor right around `swings_over`."""

    def __init__(self, swings_over=None):
        self.open = False
        self.swings_over = swings_over

    def get_state(self, fixture):
        return UNKNOWN if fixture.label != "fridge" else (OPEN if self.open else CLOSED)

    def swept_xy(self, fixture, open):
        return None if self.swings_over is None else np.array([self.swings_over])

    def set_state(self, fixture, open):
        self.open = open
        obstacle = np.array([[3.0, 4.5]]) if open else None
        return ActuationResult(True, True, "", "fridge", obstacle)


def room(size=6.0, res=0.05) -> OccupancyGrid:
    n = round(size / res)
    occupied = np.zeros((n, n), dtype=bool)
    occupied[[0, -1], :] = occupied[:, [0, -1]] = True
    return OccupancyGrid(occupied, res, (0.0, 0.0))


def make(arm_q=READY, cfg=CFG, actuator=None, **env_kwargs):
    env = FakeEnv(START, arm_q=arm_q, tcp_world=tcp_world, **env_kwargs)
    robot = Robot(env, room(), cfg, grasp_proposer=OneGrasp(), placement_proposer=OnePlacement(), fixture_actuator=actuator)
    return robot, env


def controller(arm_q=READY, cfg=CFG):
    env = FakeEnv(START, arm_q=arm_q)
    return ArmController(env, IK, cfg), env


# -- IK ------------------------------------------------------------------------


def test_ik_solves_a_reachable_pose_within_limits():
    target = pose_at([0.5, 0.1, 0.3])
    q = IK.solve_pose(target)
    assert q is not None and IK.within_limits(q, margin=1e-6)
    reached = IK.fk(q)
    assert np.linalg.norm(reached[:3, 3] - target[:3, 3]) <= CFG.ik_pose_pos_tolerance
    assert reached[:3, :3] == pytest.approx(target[:3, :3], abs=0.03)


def test_ik_rejects_an_unreachable_pose():
    assert IK.solve_pose(pose_at([1.6, 0.0, 0.3])) is None
    assert IK.solve_pose(pose_at([0.5, 0.0, -0.6])) is None  # gripper down, far below the arm base


def test_ik_stays_near_its_seed():
    start = IK.solve_pose(pose_at([0.5, 0.0, 0.3]))
    nearby = IK.solve_pose(pose_at([0.5, 0.0, 0.31]), seed=start)
    assert np.abs(nearby - start).max() < 0.1


# -- ArmController -------------------------------------------------------------


def test_joint_move_reaches_target_within_limits():
    arm, env = controller(arm_q=STOW)
    result = arm.move_joints(READY, threading.Event())
    assert result.ok
    assert np.abs(env.arm_q - READY).max() <= CFG.joint_tolerance
    times, targets = zip(*env.arm_targets)
    velocity = np.abs(np.diff(targets, axis=0)) / np.diff(times)[:, None]
    assert velocity.max() <= CFG.max_joint_vel + 1e-6
    acceleration = np.abs(np.diff(velocity, axis=0)) / CFG.control_dt
    assert acceleration.max() <= CFG.max_joint_acc * 1.15  # discretization


def test_joint_move_rejects_a_goal_outside_the_limits():
    arm, env = controller()
    goal = READY.copy()
    goal[3] = 0.5  # the elbow only bends one way
    assert arm.move_joints(goal, threading.Event()).status == MotionStatus.UNREACHABLE
    assert env.arm_targets == []


def test_cartesian_line_is_straight():
    arm, env = controller()
    start = IK.fk(READY)
    goal = start.copy()
    goal[:3, 3] += [0.10, 0.05, -0.20]
    assert arm.move_line(goal, threading.Event()).ok
    direction = (goal[:3, 3] - start[:3, 3]) / np.linalg.norm(goal[:3, 3] - start[:3, 3])
    for _, target in env.arm_targets:
        pose = IK.fk(target)
        offset = pose[:3, 3] - start[:3, 3]
        assert np.linalg.norm(offset - (offset @ direction) * direction) <= 0.003  # cart_tol
        assert pose[:3, :3] == pytest.approx(start[:3, :3], abs=0.03)
    assert np.linalg.norm(IK.fk(env.arm_q)[:3, 3] - goal[:3, 3]) <= 0.005


def test_cartesian_line_without_a_solution_is_no_ik_and_does_not_move():
    arm, env = controller()
    goal = IK.fk(READY)
    goal[:3, 3] += [0.9, 0.0, 0.0]
    result = arm.move_line(goal, threading.Event())
    assert result.status == MotionStatus.UNREACHABLE and "no_ik" in result.message
    assert env.arm_targets == []


def test_contact_mid_move_is_collision_and_the_arm_stops_within_one_tick():
    arm, env = controller(arm_q=STOW)
    env.contact_when = lambda e: e.ticks >= 8
    result = arm.move_joints(READY, threading.Event())
    assert result.status == MotionStatus.COLLISION
    stopped_at = env.arm_q.copy()
    assert env.ticks <= 9
    assert env.arm_target == pytest.approx(stopped_at)  # holding where it is, not chasing the old target
    for _ in range(5):
        env.wait_for_next_tick(CFG.control_dt)
    assert env.arm_q == pytest.approx(stopped_at)


def test_contact_is_ignored_in_a_contact_expected_phase():
    arm, env = controller(arm_q=STOW)
    env.contact_when = lambda e: e.ticks >= 8
    assert arm.move_joints(READY, threading.Event(), expect_contact=True).ok


def test_arm_can_leave_a_contact_it_starts_in():
    arm, env = controller(arm_q=STOW)
    env.arm_in_contact = True
    env.contact_when = lambda e: e.ticks < 4
    assert arm.move_joints(READY, threading.Event()).ok


def test_a_blocked_arm_is_reported_as_collision():
    arm, env = controller(arm_q=STOW)
    env.set_arm_joint_targets = lambda q: None  # the servo no longer follows
    result = arm.move_joints(READY, threading.Event())
    assert result.status == MotionStatus.COLLISION and "blocked" in result.message


# -- pick ----------------------------------------------------------------------


def test_pick_grasps_lifts_and_reports_what_is_held():
    robot, env = make()
    egg = FakeObject(OBJECT_AT.copy(), width=0.045)
    env.objects.append(egg)
    robot.pick(Detection("egg", OBJECT_AT))
    result = robot.wait()
    assert result.status == MotionStatus.SUCCESS, result.message
    held = robot.get_held()
    assert held is not None and held.label == "egg" and held.grasp_width == pytest.approx(0.045, abs=1e-3)
    assert egg.held and egg.position[2] > OBJECT_AT[2] + 0.1  # lifted
    assert np.abs(env.arm_q - READY).max() <= CFG.joint_tolerance
    assert env.gripper_commands == [False, True]  # opened for the approach, then closed
    assert np.all(env.velocity == 0.0)  # the base never moved


def test_pick_on_nothing_is_grasp_failed_and_reopens_the_gripper():
    robot, env = make()
    robot.pick(Detection("egg", OBJECT_AT))
    result = robot.wait()
    assert result.status == MotionStatus.GRASP_FAILED
    assert robot.get_held() is None
    assert not env.gripper_closed and env.gripper_width == pytest.approx(GRIPPER_OPEN)
    assert np.abs(env.arm_q - READY).max() <= CFG.joint_tolerance  # backed out


def test_pick_error_paths():
    robot, env = make()
    robot.pick(Detection("counter", OBJECT_AT))
    assert robot.wait().status == MotionStatus.NO_GRASP
    robot.pick(Detection("egg", OBJECT_AT + [1.5, 0.0, 0.0]))
    far = robot.wait()
    assert far.status == MotionStatus.UNREACHABLE and "align" in far.message
    assert env.arm_targets == []  # nothing moved for either

    env.objects.append(FakeObject(OBJECT_AT.copy()))
    robot.pick(Detection("egg", OBJECT_AT))
    assert robot.wait().status == MotionStatus.SUCCESS
    robot.pick(Detection("mug", OBJECT_AT))
    assert robot.wait().status == MotionStatus.ALREADY_HOLDING
    assert robot.get_held().label == "egg"


def test_stop_during_a_pick_halts_the_arm():
    robot, env = make(arm_q=STOW, tick_sleep=0.002)
    env.objects.append(FakeObject(OBJECT_AT.copy()))
    robot.pick(Detection("egg", OBJECT_AT))
    time.sleep(0.05)
    assert robot.get_status().status == MotionStatus.MOVING
    robot.stop()
    result = robot.get_status()
    assert result.status == MotionStatus.STOPPED
    halted = env.arm_q.copy()
    assert env.arm_target == pytest.approx(halted)
    for _ in range(5):
        env.wait_for_next_tick(CFG.control_dt)
    assert env.arm_q == pytest.approx(halted)
    assert robot.get_held() is None


def test_collision_during_a_pick_stops_and_opens_nothing():
    robot, env = make(arm_q=STOW)
    env.objects.append(FakeObject(OBJECT_AT.copy()))
    env.contact_when = lambda e: e.ticks >= 5
    robot.pick(Detection("egg", OBJECT_AT))
    assert robot.wait().status == MotionStatus.COLLISION
    assert env.gripper_commands == []


class BoxedGrasp(OneGrasp):
    """OneGrasp, with the object's box around the detection."""

    def propose(self, target):
        grasps = super().propose(target)
        for grasp in grasps:
            grasp.object_box = np.array([target.position_world - 0.03, target.position_world + 0.03])
        return grasps


class LookingDown(FakeEnv):
    """A camera straight above OBJECT_AT: it sees the object's top while the object is
    there (always, if `stays`: the fingers caught something else) and the counter once it is gone."""

    stays = False

    def get_camera_rgbd(self, cam):
        there = self.stays or any(np.linalg.norm(o.position - OBJECT_AT) < 0.01 for o in self.objects)
        top = OBJECT_AT[2] if there else OBJECT_AT[2] - 0.025
        k = np.array([[500.0, 0.0, 4.5], [0.0, 500.0, 4.5], [0.0, 0.0, 1.0]])
        return RGBD(np.zeros((10, 10, 3), np.uint8), np.full((10, 10), 2.0 - top, np.float32), k,
                    pose_at([OBJECT_AT[0], OBJECT_AT[1], 2.0]))  # fmt: skip


def seeing_robot(stays: bool, width: float):
    env = LookingDown(START, arm_q=READY, tcp_world=tcp_world)
    env.stays = stays
    env.objects.append(FakeObject(OBJECT_AT.copy(), width=width))
    cfg = replace(CFG, planning_cameras=("above",))
    return Robot(env, room(), cfg, grasp_proposer=BoxedGrasp(), placement_proposer=OnePlacement()), env


def test_pick_of_a_thin_object_that_came_up_is_held():
    robot, env = seeing_robot(stays=False, width=0.015)  # a spoon, lying flat
    robot.pick(Detection("spoon", OBJECT_AT))
    result = robot.wait()
    assert result.status == MotionStatus.SUCCESS, result.message
    assert robot.get_held() is not None and env.objects[0].held


def test_pick_that_left_the_object_where_it_was_is_grasp_failed():
    robot, env = seeing_robot(stays=True, width=0.015)  # closed on something, but the spoon is still seen there
    robot.pick(Detection("spoon", OBJECT_AT))
    result = robot.wait()
    assert result.status == MotionStatus.GRASP_FAILED and "still where it was" in result.message
    assert robot.get_held() is None
    assert not env.gripper_closed and env.gripper_width == pytest.approx(GRIPPER_OPEN)


def test_pick_closed_far_narrower_than_the_object_is_grasp_failed_when_the_camera_cannot_tell():
    robot, env = make()  # no camera
    env.objects.append(FakeObject(OBJECT_AT.copy(), width=0.013))  # an edge, where a 5 cm egg was expected
    robot.pick(Detection("egg", OBJECT_AT))
    result = robot.wait()
    assert result.status == MotionStatus.GRASP_FAILED and "narrower than the egg" in result.message
    assert robot.get_held() is None and not env.gripper_closed


# -- place ---------------------------------------------------------------------


def holding_robot(**kwargs):
    robot, env = make(**kwargs)
    egg = FakeObject(OBJECT_AT.copy(), width=0.045)
    env.objects.append(egg)
    robot.pick(Detection("egg", OBJECT_AT))
    assert robot.wait().status == MotionStatus.SUCCESS
    return robot, env, egg


def test_place_sets_the_object_down_and_lets_go():
    robot, env, egg = holding_robot()
    spot = OBJECT_AT + [0.0, 0.15, 0.0]
    robot.place(Detection("counter", spot))
    result = robot.wait()
    assert result.status == MotionStatus.SUCCESS, result.message
    assert robot.get_held() is None and not egg.held
    assert egg.position == pytest.approx(spot, abs=0.005)
    assert env.gripper_width == pytest.approx(GRIPPER_OPEN)
    assert np.abs(env.arm_q - READY).max() <= CFG.joint_tolerance


def test_place_error_paths():
    robot, env = make()
    robot.place(Detection("counter", OBJECT_AT))
    assert robot.wait().status == MotionStatus.NOT_HOLDING

    robot, env, egg = holding_robot()
    robot.place(Detection("sink", OBJECT_AT))
    assert robot.wait().status == MotionStatus.NO_PLACEMENT
    robot.place(Detection("counter", OBJECT_AT + [1.5, 0.0, 0.0]))
    far = robot.wait()
    assert far.status == MotionStatus.UNREACHABLE and "align" in far.message
    assert robot.get_held() is not None and egg.held  # still in the gripper after both


def test_place_with_the_fingers_shut_on_nothing_is_not_holding():
    robot, env, egg = holding_robot()
    egg.held, egg.width, env.gripper_width = False, 0.0, 0.0  # it slipped out unnoticed
    env.objects.clear()
    robot.place(Detection("counter", OBJECT_AT + [0.0, 0.15, 0.0]))
    assert robot.wait().status == MotionStatus.NOT_HOLDING
    assert robot.get_held() is None


# -- interlocks ----------------------------------------------------------------


def test_base_motion_tucks_the_arm_first():
    robot, env = make(arm_q=READY)
    robot.go_to_pose(Pose2D(4.5, 3.0, 0.0))
    assert robot.wait().status == MotionStatus.SUCCESS
    first_drive = next(t for t, command in env.commands if np.any(command != 0.0))
    last_arm = max(t for t, _ in env.arm_targets)
    assert last_arm < first_drive, "the arm and the base moved at the same time"
    assert np.abs(env.arm_q - STOW).max() <= CFG.joint_tolerance


def test_base_motion_while_holding_goes_to_carry_first():
    cfg = replace(CFG, arm_carry=(0.0, -1.0, 0.0, -2.4, 0.0, 1.4, 0.785))
    robot, env, egg = holding_robot(cfg=cfg)
    robot.go_to_pose(Pose2D(4.5, 3.0, 0.0))
    assert robot.wait().status == MotionStatus.SUCCESS
    assert np.abs(env.arm_q - np.array(cfg.arm_carry)).max() <= cfg.joint_tolerance
    first_drive = next(t for t, command in env.commands if np.any(command != 0.0))
    assert max(t for t, _ in env.arm_targets) < first_drive
    assert robot.get_held() is not None and egg.held  # carried along
    assert egg.position[:2] == pytest.approx(tcp_world(env.pose, env.arm_q)[:2], abs=1e-6)


def test_dropping_the_object_while_driving_is_grasp_failed():
    robot, env, egg = holding_robot(tick_sleep=0.001)
    robot.go_to_pose(Pose2D(4.8, 3.0, 0.0))
    while not any(np.any(command != 0.0) for _, command in env.commands):
        time.sleep(0.005)
    egg.held, egg.width = False, 0.0  # it slips out; the fingers close on nothing
    env.objects.clear()
    result = robot.wait()
    assert result.status == MotionStatus.GRASP_FAILED and "dropped" in result.message
    assert robot.get_held() is None
    assert np.all(env.commands[-1][1] == 0.0)


def test_arm_skills_refuse_while_the_base_is_moving():
    robot, env = make()
    env.objects.append(FakeObject(OBJECT_AT.copy()))
    env.velocity = np.array([0.2, 0.0, 0.0])  # coasting, not under the robot layer's control
    robot.pick(Detection("egg", OBJECT_AT))
    result = robot.wait()
    assert result.status == MotionStatus.BLOCKED and "moving" in result.message
    assert env.arm_targets == [] and env.gripper_commands == []


def test_a_new_skill_preempts_the_motion_in_progress():
    robot, env = make(tick_sleep=0.002)
    env.objects.append(FakeObject(OBJECT_AT.copy()))
    robot.go_to_pose(Pose2D(1.0, 1.0, 0.0))
    time.sleep(0.05)
    robot.pick(Detection("counter", OBJECT_AT))  # stops the drive, then fails fast
    assert robot.wait().status == MotionStatus.NO_GRASP
    assert np.all(env.velocity == 0.0)


# -- actuate_fixture -----------------------------------------------------------

FRIDGE = Detection("fridge", np.array([3.7, 3.0, 1.0]))


def test_actuate_opens_and_closes_and_says_it_is_scripted():
    door = Door()
    robot, env = make(actuator=door)
    robot.actuate_fixture(FRIDGE, True)
    result = robot.wait()
    assert result.status == MotionStatus.SUCCESS and result.data == {"scripted": True}
    assert door.open
    robot.actuate_fixture(FRIDGE, True)
    assert "already open" in robot.wait().message
    robot.actuate_fixture(FRIDGE, False)
    assert robot.wait().status == MotionStatus.SUCCESS and not door.open


def test_actuate_error_paths():
    robot, env = make(actuator=Door())
    robot.actuate_fixture(Detection("egg", FRIDGE.position_world), True)
    assert robot.wait().status == MotionStatus.UNSUPPORTED
    robot.actuate_fixture(Detection("fridge", np.array([5.5, 3.0, 1.0])), True)
    far = robot.wait()
    assert far.status == MotionStatus.UNREACHABLE and "align" in far.message
    robot_without, _ = make()
    robot_without.actuate_fixture(FRIDGE, True)
    assert robot_without.wait().status == MotionStatus.UNSUPPORTED


def test_robot_steps_out_of_a_doors_way_before_it_moves():
    door = Door(swings_over=(START.x, START.y))
    robot, env = make(actuator=door)
    robot.actuate_fixture(FRIDGE, True)
    result = robot.wait()
    assert result.status == MotionStatus.SUCCESS and door.open
    moved = math.hypot(env.pose.x - START.x, env.pose.y - START.y)
    assert moved > CFG.inflation_radius, "the base is still inside the door's arc"
    assert np.abs(env.arm_q - STOW).max() <= CFG.joint_tolerance  # tucked for the step back


def test_an_open_door_is_an_obstacle_until_it_is_closed():
    door = Door()
    robot, env = make(actuator=door)
    through_the_door = Pose2D(3.0, 4.5, 0.0)
    robot.actuate_fixture(FRIDGE, True)
    assert robot.wait().status == MotionStatus.SUCCESS
    robot.go_to_pose(through_the_door)
    snapped = robot.wait()
    assert snapped.status == MotionStatus.SUCCESS
    assert snapped.final_pose.distance_to(through_the_door) >= CFG.inflation_radius - 0.05  # goal moved off the door
    robot.go_to_pose(START)
    robot.wait()
    robot.actuate_fixture(FRIDGE, False)
    assert robot.wait().status == MotionStatus.SUCCESS
    robot.go_to_pose(through_the_door)
    assert robot.wait().final_pose.distance_to(through_the_door) <= 0.03
