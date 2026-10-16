"""The arm's capsule model, the depth-image world, and joint paths around it."""

import numpy as np
import pytest

pytest.importorskip("coal")
pytest.importorskip("ompl")

from chefbot.interface.types import RGBD, Pose2D  # noqa: E402
from chefbot.robot.arm.ik import arm_ik_from_config  # noqa: E402
from chefbot.robot.config import RobotConfig  # noqa: E402
from chefbot.robot.kinematics.collision import (  # noqa: E402
    ArmCollisionModel,
    Box,
    Capsule,
    Workspace,
    depth_points,
    filter_points,
)

CFG = RobotConfig.load()
READY = np.array(CFG.arm_ready)


@pytest.fixture(scope="module")
def ik():
    return arm_ik_from_config(CFG)


@pytest.fixture(scope="module")
def model(ik):
    return ArmCollisionModel(ik.model)


def wall(x: float, y=(-0.6, 0.6), z=(-0.4, 0.9), step=0.01) -> np.ndarray:
    """A wall of points across the arm's front at distance `x` (arm base frame)."""
    ys, zs = np.meshgrid(np.arange(*y, step), np.arange(*z, step))
    return np.stack([np.full(ys.size, x), ys.ravel(), zs.ravel()], axis=1)


def test_the_arm_is_a_chain_of_capsules_ending_in_the_gripper_at_the_tcp(model, ik):
    shapes = model.shapes(READY)
    links = [s for s in shapes if isinstance(s, Capsule)]
    for a, b in zip(links, links[1:]):
        assert a.p1 == pytest.approx(b.p0, abs=1e-6)  # one connected chain
    tcp = ik.fk(READY)
    pads = shapes[-1]
    assert isinstance(pads, Box) and pads.distance(tcp[:3, 3][None])[0] == 0.0  # the TCP is between the pads
    assert pads.rotation == pytest.approx(tcp[:3, :3])  # boxes are aligned with the TCP frame
    assert len(model.shapes(READY, pads=False)) == len(shapes) - 1


def test_without_a_world_everything_is_free_and_straight():
    world = Workspace.unchecked()
    assert not world.checked and world.free(READY)
    assert [list(q) for q in world.path(READY, -READY)] == [list(READY), list(-READY)]


def test_a_wall_in_front_blocks_reaching_through_it(model, ik):
    world = Workspace(model, wall(0.55), joint_limits=ik.joint_limits)
    assert world.checked
    assert world.free(READY)  # the ready pose is short of the wall
    reach = ik.solve_position(np.array([0.75, 0.0, 0.3]))  # past the wall
    assert reach is not None and not world.free(reach)
    assert world.path(READY, reach) is None  # the goal itself is in collision


def test_a_path_is_planned_around_an_obstacle_the_straight_move_would_hit(model, ik):
    from ompl import util

    util.RNG.setSeed(7)  # RRT is random: a fixed seed keeps the test's run time steady
    start = ik.solve_pose(_down_at(0.45, 0.35, 0.25), seed=READY)
    goal = ik.solve_pose(_down_at(0.45, -0.35, 0.25), seed=READY)
    assert start is not None and goal is not None
    # A post between the two, as tall as the arm swings low.
    ys, zs = np.meshgrid(np.arange(-0.06, 0.06, 0.01), np.arange(-0.4, 0.45, 0.01))
    post = np.concatenate([np.stack([np.full(ys.size, x), ys.ravel(), zs.ravel()], axis=1)
                           for x in np.arange(0.35, 0.56, 0.02)])  # fmt: skip
    world = Workspace(model, post, joint_limits=ik.joint_limits, timeout_s=10.0)
    assert world.free(start) and world.free(goal)
    assert not world.segment_free(start, goal)
    path = world.path(start, goal)
    assert path is not None and len(path) > 2
    assert path[0] == pytest.approx(start) and path[-1] == pytest.approx(goal)
    assert all(world.segment_free(a, b) for a, b in zip(path, path[1:]))


def test_depth_image_back_projects_onto_the_wall_it_shows():
    # A camera 1 m from a wall, looking straight at it (OpenCV frame: z forward).
    k = np.array([[300.0, 0.0, 32.0], [0.0, 300.0, 24.0], [0.0, 0.0, 1.0]])
    cam_to_world = np.eye(4)
    cam_to_world[:3, :3] = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]  # looking along world +x
    cam_to_world[:3, 3] = [1.0, 2.0, 1.5]
    depth = np.full((48, 64), 1.0, np.float32)
    depth[0, 0] = 0.0  # no reading
    points = depth_points(RGBD(np.zeros((48, 64, 3), np.uint8), depth, k, cam_to_world))
    assert len(points) == 48 * 64 - 1
    assert points[:, 0] == pytest.approx(2.0)  # all on the wall, 1 m in front
    center = np.argmin(np.linalg.norm(points[:, 1:] - [2.0, 1.5], axis=1))
    assert points[center] == pytest.approx([2.0, 2.0, 1.5], abs=1e-2)


def test_the_arm_the_base_the_floor_and_ignored_boxes_are_not_obstacles(model):
    from chefbot.robot.kinematics.reachability import ReachabilityChecker

    base = Pose2D(1.0, 2.0, 0.5)
    arm_in_world = ReachabilityChecker(CFG).arm_base_in_world(base)
    on_arm = [c.p0 + t * (c.p1 - c.p0) for c in model.shapes(READY) if isinstance(c, Capsule)
              for t in np.linspace(0, 1, 5)]  # fmt: skip
    on_arm = [arm_in_world[:3, :3] @ p + arm_in_world[:3, 3] for p in on_arm]
    on_base, on_floor = [1.1, 2.1, 0.5], [2.0, 3.0, 0.0]
    cup, shelf = np.array([2.2, 2.6, 0.9]), np.array([2.2, 2.6, 1.6])
    kept = filter_points(
        np.array([*on_arm, on_base, on_floor, cup, shelf]), base, arm_in_world, READY, model,
        CFG.self_filter_margin, CFG.base_box, [np.array([cup - 0.05, cup + 0.05])],
    )  # fmt: skip
    kept_world = kept @ arm_in_world[:3, :3].T + arm_in_world[:3, 3]
    assert kept_world == pytest.approx(shelf[None], abs=1e-9)  # only the shelf is an obstacle


def _down_at(x, y, z):
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])  # tool z pointing down
    pose[:3, 3] = [x, y, z]
    return pose
