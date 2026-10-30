"""Acceptance tests from NAV_SPEC.md section 12, run in RoboCasa on two layouts,
in-process and over RemoteRobot."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from chefbot.agent.config import AgentConfig
from chefbot.interface.types import Pose2D
from chefbot.robot.kinematics.reachability import ReachabilityChecker
from chefbot.scripts.demo_nav import TRACE, run_trace

pytestmark = pytest.mark.robocasa

AGENT = AgentConfig.load()


def ids(result, key="detections"):
    return [item["id"] for item in result.data[key]]


def ground_truth(stack, label: str) -> list[np.ndarray]:
    """True world positions of every instance of `label`, visible or not."""
    detector = stack.detector
    return [detector._position(item) for item in detector._items if item.label == label]


def test_go_to_every_fixture_from_three_random_free_starts(session):
    stack = session.stack
    contacts_before = stack.env.contact_ticks
    for start in stack.free_starts(3, seed=stack.layout):
        for entry in stack.world_map.list():
            stack.env.teleport_base(start)
            result = session.call("go_to", place=entry.name)
            assert result.success, f"{entry.name} from {start}: {result.message}"
            pose = stack.env.get_state().base_pose
            # Near the fixture and facing it (coarse navigation: +-10 degrees).
            distance = math.hypot(entry.location[0] - pose.x, entry.location[1] - pose.y)
            assert distance <= AGENT.approach_distance + stack.robot_config.snap_radius + 0.1, entry.name
            bearing = math.atan2(entry.location[1] - pose.y, entry.location[0] - pose.x)
            if entry.facing is not None:
                assert abs(math.remainder(bearing - pose.yaw, math.tau)) <= math.radians(10), entry.name
    assert stack.env.contact_ticks == contacts_before, "the robot touched something on the way"


@pytest.mark.parametrize(
    "place, part",
    [("stove", "burner"), ("fridge", "fridge handle"), ("microwave", "microwave handle")],
)
def test_go_to_detect_nearby_align(session, place, part):
    stack = session.stack
    contacts_before = stack.env.contact_ticks
    stack.env.teleport_base(stack.free_starts(1, seed=7)[0])

    assert session.call("go_to", place=place).success
    detected = session.call("detect", classes=[part])
    assert detected.success and ids(detected), f"no {part} seen from in front of the {place}"
    assert all(item["label"] == part for item in detected.data["detections"])
    view = session.call("nearby")
    assert set(ids(detected)) <= set(ids(view, "visible"))
    assert place in ids(view, "visible")  # the fixture itself was matched to the map

    target_id = ids(detected)[0]
    result = session.call("align", id=target_id)
    assert result.success, result.message
    assert result.data["refined"] is True

    # Final reach verified by IK, against the true part position and the true base pose.
    pose = stack.env.get_state().base_pose
    stored = session.registry.tools.detections.get(target_id).position_world
    truth = min(ground_truth(stack, part), key=lambda p: np.linalg.norm(p - stored))
    assert np.linalg.norm(truth - stored) < 0.02
    assert ReachabilityChecker(stack.robot_config).is_reachable(pose, truth)
    assert stack.grid.inflate(stack.robot_config.footprint_radius).is_free(pose.x, pose.y)
    assert stack.env.contact_ticks == contacts_before


def test_map_entry_out_of_view_is_not_visible_and_not_alignable(session):
    stack = session.stack
    stack.env.teleport_base(stack.free_starts(1, seed=3)[0])
    assert session.call("go_to", place="stove").success
    session.call("detect", classes=["burner"])
    view = session.call("nearby")
    assert "stove" in ids(view, "visible")
    hidden = ids(view, "not_visible")
    assert hidden, "expected at least one nearby map entry outside the camera's view"
    if stack.layout == 2:
        assert "sink" in hidden  # on the island, directly behind the robot
    pose = stack.env.get_state().base_pose
    for name in hidden:
        result = session.call("align", id=name)
        assert (result.success, result.error) == (False, "not_visible"), name
        assert f'go_to("{name}") first' in result.message
    assert stack.env.get_state().base_pose == pose  # none of that moved the base


def test_go_to_without_waiting_then_stop_halts_the_base(session, monkeypatch):
    stack = session.stack
    stove, fridge = stack.world_map.get("stove"), stack.world_map.get("fridge")
    assert session.call("go_to", place="stove").success
    start = stack.env.get_state().base_pose

    # The simulator runs many times faster than real time: without this the
    # whole drive can be over before `stop` is called.
    tick = stack.env.wait_for_next_tick
    monkeypatch.setattr(stack.env, "wait_for_next_tick", lambda dt: (time.sleep(0.01), tick(dt))[1])
    started = session.call("go_to", place="fridge", wait=False)
    assert started.success and started.data["status"] == "moving"
    assert session.call("wait", timeout_s=0.3).data["status"] == "moving"
    assert session.call("stop").success

    status = session.call("get_status")
    assert (status.success, status.error) == (False, "stopped"), status
    state = stack.env.get_state()
    assert np.linalg.norm(state.base_velocity) < 0.02
    travelled = state.base_pose.distance_to(start)
    total = math.hypot(fridge.location[0] - stove.location[0], fridge.location[1] - stove.location[1])
    assert 0.02 < travelled < total - 0.3, "the base should have set off and stopped well short"
    # And it stays put: a later wait reports the same stopped motion at the same pose.
    assert session.call("wait").error == "stopped"
    assert stack.env.get_state().base_pose == state.base_pose


def test_demo_trace(session):
    stack = session.stack
    contacts_before = stack.env.contact_ticks
    stack.env.teleport_base(stack.free_starts(1, seed=11)[0])
    results = run_trace(session.registry, verbose=False)
    for (name, args), result in zip(TRACE, results):
        assert result.success, f"{name}({args}): {result.message}"
    assert stack.env.contact_ticks == contacts_before
