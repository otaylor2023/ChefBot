"""RoboCasaEnv, the map builder and the ground-truth detector on their own."""

from __future__ import annotations

import math

import numpy as np
import pytest

from chefbot.interface.types import Pose2D

pytestmark = pytest.mark.robocasa


def run(env, command, seconds, frame="world"):
    """Hold a velocity command; returns (average velocity, final state)."""
    before = env.get_state()
    env.set_base_velocity(*command, frame=frame)
    for _ in range(round(seconds * env.config.control_freq)):
        env.wait_for_next_tick(1.0 / env.config.control_freq)
    after = env.get_state()
    elapsed = after.time - before.time
    average = np.array(
        [
            (after.base_pose.x - before.base_pose.x) / elapsed,
            (after.base_pose.y - before.base_pose.y) / elapsed,
            math.remainder(after.base_pose.yaw - before.base_pose.yaw, math.tau) / elapsed,
        ]
    )
    return average, after


@pytest.fixture
def env(stack):
    stack.robot.stop_base()
    stack.env.teleport_base(stack.free_starts(1, seed=5)[0])
    yield stack.env
    stack.env.set_base_velocity(0.0, 0.0, 0.0)


def test_teleport_places_the_base_at_rest(env):
    target = Pose2D(env.get_state().base_pose.x, env.get_state().base_pose.y, 2.0)
    env.teleport_base(target)
    state = env.get_state()
    assert state.base_pose.distance_to(target) < 1e-6
    assert abs(state.base_pose.yaw - 2.0) < 1e-6
    assert np.allclose(state.base_velocity, 0.0) and not state.base_in_contact
    _, later = run(env, (0.0, 0.0, 0.0), 0.5)
    assert later.base_pose.distance_to(target) < 2e-3  # and it stays there
    assert later.time == pytest.approx(state.time + 0.5)


@pytest.mark.parametrize("yaw", [0.0, 1.0, -2.5])
def test_base_is_holonomic_and_tracks_world_frame_velocity(env, yaw):
    pose = env.get_state().base_pose
    env.teleport_base(Pose2D(pose.x, pose.y, yaw))
    for command in [(0.2, 0.0, 0.0), (0.0, -0.2, 0.0), (0.1, 0.1, 0.0), (0.0, 0.0, 0.5), (0.03, 0.0, 0.0)]:
        run(env, command, 0.3)  # get up to speed
        average, state = run(env, command, 0.5)
        assert average == pytest.approx(command, abs=0.02), command
        assert state.base_velocity == pytest.approx(command, abs=0.02), command
        run(env, (-command[0], -command[1], -command[2]), 0.8)  # and back, to stay in free space
    run(env, (0.0, 0.0, 0.0), 0.2)


def test_base_frame_commands_follow_the_heading(env):
    pose = env.get_state().base_pose
    env.teleport_base(Pose2D(pose.x, pose.y, 2.0))
    run(env, (0.2, 0.0, 0.0), 0.3, frame="base")
    average, _ = run(env, (0.2, 0.0, 0.0), 0.3, frame="base")
    assert average[:2] == pytest.approx([0.2 * math.cos(2.0), 0.2 * math.sin(2.0)], abs=0.02)
    with pytest.raises(ValueError):
        env.set_base_velocity(0.1, 0.0, 0.0, frame="odom")


def test_zero_command_stops_the_base(env):
    run(env, (0.3, 0.0, 0.0), 0.5)
    _, stopped = run(env, (0.0, 0.0, 0.0), 0.2)
    assert np.linalg.norm(stopped.base_velocity) < 0.01
    _, later = run(env, (0.0, 0.0, 0.0), 0.5)
    assert later.base_pose.distance_to(stopped.base_pose) < 2e-3


def test_contact_is_reported_when_driving_into_a_fixture(stack, env):
    stove = stack.world_map.get("stove")
    location, facing = np.array(stove.location), np.array(stove.facing)
    start = location + 1.2 * facing
    env.teleport_base(Pose2D(start[0], start[1], math.atan2(-facing[1], -facing[0])))
    assert not env.get_state().base_in_contact
    ticks = env.contact_ticks
    env.set_base_velocity(*(-0.3 * facing), 0.0)
    for _ in range(100):
        env.wait_for_next_tick(0.05)
        if env.get_state().base_in_contact:
            break
    else:
        pytest.fail("drove at the stove for 5 s without touching it")
    assert env.contact_ticks > ticks
    env.contact_ticks = ticks  # deliberate collision: do not count it against other tests


def test_grid_matches_the_scene(stack):
    grid, env = stack.grid, stack.env
    assert grid.resolution == env.config.grid_resolution
    assert not grid.is_free(-50.0, -50.0)
    inflated = grid.inflate(stack.robot_config.inflation_radius)
    assert 0.05 < (~inflated.occupied).mean() < 0.9
    for entry in stack.world_map.list():
        assert not grid.is_free(*entry.location), f"{entry.name} should sit on occupied cells"
        if entry.facing is not None:
            ahead = np.array(entry.location) + 1.0 * np.array(entry.facing)
            assert grid.is_free(*ahead), f"floor in front of {entry.name} should be free"


def test_semantic_map_uses_chefbot_names_only(stack):
    names = [e.name for e in stack.world_map.list()]
    for required in ("stove", "fridge", "microwave", "sink"):
        assert required in names
    assert len([n for n in names if n.startswith("counter")]) >= 2
    for name in names:
        assert "main_group" not in name and name == name.lower()
    # Parts are found by detection, not stored on the map.
    assert not any("burner" in n or "handle" in n for n in names)
    assert stack.world_map.labels() <= stack.detector.labels()


def test_detector_only_reports_what_the_camera_can_see(stack, env):
    stove = stack.world_map.get("stove")
    location, facing = np.array(stove.location), np.array(stove.facing)
    spot = location + 1.0 * facing
    toward = math.atan2(-facing[1], -facing[0])

    env.teleport_base(Pose2D(spot[0], spot[1], toward))
    seen = stack.detector.detect(["burner", "stove", "spatula"])
    labels = [d.label for d in seen]
    burners = sum(item.label == "burner" for item in stack.detector._items)
    assert burners >= 4 and labels.count("burner") == burners  # every burner, nothing made up
    assert labels.count("stove") == 1 and len(labels) == burners + 1
    for detection in seen:
        assert detection.position_world.shape == (3,) and detection.confidence == 1.0
    burner_heights = [d.position_world[2] for d in seen if d.label == "burner"]
    assert all(0.8 < z < 1.1 for z in burner_heights)

    env.teleport_base(Pose2D(spot[0], spot[1], toward + math.pi))  # back to the stove
    assert stack.detector.detect(["burner", "stove"]) == []


def test_detector_respects_occlusion(stack, env):
    """The island's dishwasher handle faces the aisle; from behind the island it
    is inside the camera frustum but hidden by the island itself."""
    if stack.layout != 2:
        pytest.skip("needs the island layout")
    dishwasher = stack.world_map.get("dishwasher")
    location, facing = np.array(dishwasher.location), np.array(dishwasher.facing)
    toward_front = math.atan2(-facing[1], -facing[0])

    front = location + 1.3 * facing
    env.teleport_base(Pose2D(front[0], front[1], toward_front))
    assert [d.label for d in stack.detector.detect(["dishwasher handle"])] == ["dishwasher handle"]

    # The nearest spot on the far side of the island where the base fits.
    footprint = stack.grid.inflate(stack.robot_config.footprint_radius)
    behind = next(
        (p for d in np.arange(0.8, 3.0, 0.05) if footprint.is_free(*(p := location - d * facing))), None
    )
    assert behind is not None, "no floor behind the island"
    env.teleport_base(Pose2D(behind[0], behind[1], toward_front + math.pi))
    handle = next(i for i in stack.detector._items if i.label == "dishwasher handle")
    # Same check with occlusion switched off: the handle is in the frustum from here.
    original = stack.detector._first_hit
    stack.detector._first_hit = lambda origin, direction: (-1, math.inf)
    try:
        in_frustum_only = stack.detector._visible(handle)
    finally:
        stack.detector._first_hit = original
    assert in_frustum_only
    assert stack.detector.detect(["dishwasher handle"]) == []


def test_get_camera_rgbd_requires_rendering(env):
    with pytest.raises(RuntimeError, match="render=True"):
        env.get_camera_rgbd(env.config.shoulder_camera)
    with pytest.raises(NotImplementedError):
        env.set_arm_joint_targets(np.zeros(7))


def test_shoulder_camera_rgbd_agrees_with_ground_truth():
    """Depth at a detected burner's pixel matches the burner's distance from the camera."""
    from chefbot.envs.robocasa.config import RoboCasaConfig
    from chefbot.envs.robocasa.detector import GroundTruthDetector, map_fixtures
    from chefbot.envs.robocasa.env import RoboCasaEnv

    env = RoboCasaEnv(RoboCasaConfig.load(layout=1, style=1), render=True)
    try:
        box = next(box for _, label, box in map_fixtures(env) if label == "stove")
        spot = box.center[:2] + 1.0 * box.facing
        env.teleport_base(Pose2D(spot[0], spot[1], math.atan2(-box.facing[1], -box.facing[0])))
        image = env.get_camera_rgbd(env.config.shoulder_camera)
        height, width = env.config.camera_height, env.config.camera_width
        assert image.rgb.shape == (height, width, 3) and image.rgb.dtype == np.uint8
        assert image.depth.shape == (height, width) and image.depth.dtype == np.float32
        assert image.rgb.std() > 10  # an actual picture, not a blank frame

        burners = GroundTruthDetector(env).detect(["burner"])
        assert len(burners) == 4
        world_to_cam = np.linalg.inv(image.cam_to_world)
        matched = 0
        for burner in burners:
            in_camera = (world_to_cam @ np.append(burner.position_world, 1.0))[:3]
            u, v, w = image.intrinsics @ in_camera
            row, col = round(v / w), round(u / w)
            assert 0 <= row < height and 0 <= col < width  # detections are inside the image
            matched += abs(image.depth[row, col] - in_camera[2]) < 0.03
        # The robot's own arm hides part of the stove; the detector looks past it.
        assert matched >= 2
    finally:
        env.close()
