import math
import threading
import time

import numpy as np
import pytest

from chefbot.agent.config import AgentConfig
from chefbot.agent.tools.registry import ToolRegistry
from chefbot.agent.world.map import WorldMap
from chefbot.interface.detector import Detector
from chefbot.interface.proposers import GraspProposer, PlacementProposer
from chefbot.interface.transport.client import RemoteDetector, RemoteError, RemoteRobot, _Channel
from chefbot.interface.transport.server import RobotServer
from chefbot.interface.types import Detection, Grasp, HeldObject, MotionStatus, Placement, Pose2D
from chefbot.robot.config import RobotConfig
from chefbot.robot.kinematics.reachability import ReachabilityChecker
from chefbot.robot.nav.grid import OccupancyGrid
from chefbot.robot.robot import Robot
from chefbot.tests.robot.fake_env import FakeEnv, FakeObject

CFG = RobotConfig(footprint_radius=0.2, safety_margin=0.05)


class Reach:
    def is_reachable(self, base_pose, target_world):
        return math.hypot(target_world[0] - base_pose.x, target_world[1] - base_pose.y) <= 0.7


class StaticDetector(Detector):
    def __init__(self):
        self.fail = False

    def detect(self, classes):
        if self.fail:
            raise RuntimeError("camera unplugged")
        objects = [("burner", (3.0, 5.6, 0.9)), ("pan", (3.2, 5.6, 0.95)), ("stove", (3.0, 5.7, 0.9))]
        return [Detection(label, np.array(pos), 1.0) for label, pos in objects if label in classes]


@pytest.fixture
def served():
    occupied = np.zeros((120, 120), dtype=bool)
    occupied[[0, -1], :] = occupied[:, [0, -1]] = True
    occupied[110:, 40:80] = True  # stove against the far wall
    env = FakeEnv(Pose2D(1.0, 1.0, 0.0))
    detector = StaticDetector()
    server = RobotServer(Robot(env, OccupancyGrid(occupied, 0.05, (0.0, 0.0)), CFG, Reach()), detector, port=0)
    server.start()
    robot, remote_detector = RemoteRobot(server.url), RemoteDetector(server.url)
    yield robot, remote_detector, env, detector, server
    robot.stop()
    robot.close()
    remote_detector.close()
    server.shutdown()


def test_remote_robot_api(served):
    robot, _, env, _, _ = served
    assert robot.get_base_pose() == Pose2D(1.0, 1.0, 0.0)
    assert robot.get_status().status == MotionStatus.IDLE
    goal = Pose2D(4.0, 3.0, 1.0)
    assert robot.go_to_pose(goal) is None
    result = robot.wait()
    assert result.status == MotionStatus.SUCCESS
    assert result.final_pose.distance_to(goal) <= CFG.pos_tolerance + 1e-3
    assert robot.get_base_pose() == env.pose
    robot.align_to_point(np.array([3.0, 5.6, 0.9]))
    assert robot.wait().status == MotionStatus.SUCCESS
    assert math.hypot(3.0 - env.pose.x, 5.6 - env.pose.y) <= 0.7


def test_remote_detector(served):
    _, detector, _, _, _ = served
    found = detector.detect(["burner", "pan", "kettle"])
    assert [d.label for d in found] == ["burner", "pan"]
    assert isinstance(found[0].position_world, np.ndarray)
    assert tuple(found[0].position_world) == (3.0, 5.6, 0.9)


def test_wait_timeout_returns_moving_over_the_wire(served):
    robot, _, env, _, _ = served
    env.tick_sleep = 0.002
    robot.go_to_pose(Pose2D(5.0, 5.0, 0.0))
    assert robot.wait(timeout_s=0.05).status == MotionStatus.MOVING
    env.tick_sleep = 0.0
    assert robot.wait().status == MotionStatus.SUCCESS


def test_stop_while_remote_wait_is_in_progress(served):
    robot, _, env, _, _ = served
    env.tick_sleep = 0.005
    robot.go_to_pose(Pose2D(5.0, 5.0, 0.0))
    waited = {}
    waiter = threading.Thread(target=lambda: waited.update(result=robot.wait()))
    waiter.start()
    time.sleep(0.15)
    assert waiter.is_alive(), "wait() should still be blocking on the server"
    robot.stop()
    waiter.join(timeout=5.0)
    assert not waiter.is_alive()
    assert waited["result"].status == MotionStatus.STOPPED
    assert np.all(env.commands[-1][1] == 0.0)
    assert env.pose.distance_to(Pose2D(5.0, 5.0, 0.0)) > 1.0  # it really stopped short
    ticks = env.ticks
    time.sleep(0.05)
    assert env.ticks == ticks


def test_server_errors_surface_as_remote_error(served):
    robot, detector, _, local_detector, server = served
    local_detector.fail = True
    with pytest.raises(RemoteError, match="camera unplugged"):
        detector.detect(["burner"])
    # The connection survives a failed call.
    local_detector.fail = False
    assert len(detector.detect(["burner"])) == 1
    with pytest.raises(RemoteError, match="unknown call"):
        _Channel(server.url, "robot").call("_drive")
    with pytest.raises(RemoteError, match="unknown call"):
        _Channel(server.url, "env").call("get_state")


def test_agent_tools_run_over_the_remote_pair(served):
    robot, detector, env, _, _ = served
    world_map = WorldMap.from_dict({"fixtures": {"stove": {"location": [3.0, 5.7], "facing": [0.0, -1.0]}}})
    registry = ToolRegistry(robot, detector, world_map, AgentConfig(default_vocab=("burner",)))
    assert registry.dispatch("go_to", {"place": "stove"}).success
    assert registry.dispatch("detect", {"classes": ["burner", "pan"]}).data["detections"] == [
        {"id": "burner_1", "label": "burner"},
        {"id": "pan_1", "label": "pan"},
    ]
    assert [v["id"] for v in registry.dispatch("nearby").data["visible"]] == ["burner_1", "pan_1", "stove"]
    aligned = registry.dispatch("align", {"id": "burner_1"})
    assert aligned.success and aligned.data["refined"] is True
    env.tick_sleep = 0.002
    robot.go_to_pose(Pose2D(1.0, 1.0, 0.0))
    assert robot.wait(timeout_s=0.05).status == MotionStatus.MOVING
    assert registry.dispatch("stop").success
    assert registry.dispatch("get_status").error == "stopped"


# -- manipulation over the wire --------------------------------------------------


class OneGrasp(GraspProposer):
    def propose(self, target):
        pose = np.eye(4)
        pose[:3, :3] = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        pose[:3, 3] = target.position_world
        return [Grasp(pose, 0.10, 0.05, 1.0)]


class OnePlacement(PlacementProposer):
    def propose(self, surface, held):
        pose = np.eye(4)
        pose[:3, :3] = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        pose[:3, 3] = surface.position_world
        return [Placement(pose, 0.10, 1.0)]


@pytest.fixture
def served_arm():
    config = RobotConfig(footprint_radius=0.2, safety_margin=0.05, arm_mount_xyz=(0.2, 0.0, 0.7))
    frames = ReachabilityChecker(config)
    start = Pose2D(3.0, 3.0, 0.0)
    env = FakeEnv(
        start,
        arm_q=config.arm_ready,
        tcp_world=lambda base, q: frames.pose_to_world(base, frames.ik.fk(q))[:3, 3],
    )
    occupied = np.zeros((120, 120), dtype=bool)
    occupied[[0, -1], :] = occupied[:, [0, -1]] = True
    local = Robot(
        env, OccupancyGrid(occupied, 0.05, (0.0, 0.0)), config,
        grasp_proposer=OneGrasp(), placement_proposer=OnePlacement(),
    )  # fmt: skip
    server = RobotServer(local, StaticDetector(), port=0)
    server.start()
    robot = RemoteRobot(server.url)
    yield robot, env
    robot.stop()
    robot.close()
    server.shutdown()


def test_remote_pick_place_and_get_held(served_arm):
    robot, env = served_arm
    at = np.array([3.75, 3.0, 0.95])
    egg = FakeObject(at.copy(), width=0.045)
    env.objects.append(egg)
    assert robot.get_held() is None
    assert robot.pick(Detection("egg", at)) is None
    assert robot.wait().status == MotionStatus.SUCCESS
    held = robot.get_held()
    assert isinstance(held, HeldObject) and held.label == "egg"
    assert tuple(held.picked_from) == tuple(at)
    robot.pick(Detection("mug", at))
    assert robot.wait().status == MotionStatus.ALREADY_HOLDING
    spot = at + [0.0, 0.15, 0.0]
    robot.place(Detection("counter", spot))
    assert robot.wait().status == MotionStatus.SUCCESS
    assert robot.get_held() is None and egg.position == pytest.approx(spot, abs=0.005)


def test_remote_stop_halts_the_arm_mid_pick(served_arm):
    robot, env = served_arm
    env.tick_sleep = 0.003
    env.objects.append(FakeObject(np.array([3.75, 3.0, 0.95])))
    robot.pick(Detection("egg", np.array([3.75, 3.0, 0.95])))
    assert robot.wait(timeout_s=0.05).status == MotionStatus.MOVING
    robot.stop()
    assert robot.get_status().status == MotionStatus.STOPPED
    halted = env.arm_q.copy()
    time.sleep(0.05)
    assert env.arm_q == pytest.approx(halted)


def test_remote_actuate_without_an_actuator_is_unsupported(served_arm):
    robot, _ = served_arm
    robot.actuate_fixture(Detection("fridge", np.array([3.7, 3.0, 1.0])), True)
    result = robot.wait()
    assert result.status == MotionStatus.UNSUPPORTED and result.data == {}
