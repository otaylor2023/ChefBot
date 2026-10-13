"""Manipulation in RoboCasa through the agent tools (MANIP_SPEC §11, integration)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from chefbot.agent.config import AgentConfig
from chefbot.agent.tools.registry import ToolRegistry
from chefbot.envs.robocasa.detector import named_fixtures, scene_objects
from chefbot.envs.robocasa.objects import object_points
from chefbot.interface.fixtures import CLOSED, OPEN, UNKNOWN
from chefbot.interface.types import Detection, Pose2D, ToolResult
from chefbot.scripts.demo_manip import manip_trace

pytestmark = pytest.mark.robocasa

# The start set that can be picked off a counter; the cutting board is a surface only.
PICKABLE = ["mug", "bowl", "bottled water", "pan"]
TRIALS = 5


def ids(result, key="detections"):
    return [item["id"] for item in result.data[key]]


def body_of(stack, label: str) -> int:
    return next(body for _, name, body in scene_objects(stack.env) if name == label)


def lowest_point(stack, label: str) -> float:
    return float(object_points(stack.env, body_of(stack, label))[:, 2].min())


def lifted(stack, label: str, start_lowest: float) -> bool:
    """Every part of the object is clear of what it stood on."""
    return lowest_point(stack, label) > start_lowest + 0.03


def fetch(call, label: str):
    """go_to -> detect -> align -> pick, as an agent would. Returns the last result."""
    result = call("go_to", place=label)
    if not result.success:
        return result
    seen = call("detect", classes=[label])
    if not ids(seen):
        return ToolResult(False, "not_seen", f"no {label} in view after go_to")
    target = ids(seen)[0]
    result = call("align", id=target)
    if not result.success:
        return result
    return call("pick", id=target)


# -- env: arm and gripper ------------------------------------------------------


def test_arm_follows_joint_targets_and_gripper_reports_width(stack):
    stack.reset()
    env, cfg = stack.env, stack.robot_config
    state = env.get_state()
    assert state.arm_q == pytest.approx(np.array(cfg.arm_stow), abs=0.02)
    assert not state.arm_in_contact and not state.base_in_contact
    assert state.gripper_width == pytest.approx(cfg.gripper_open_width, abs=0.004) and not state.gripper_closed_cmd

    target = np.array(cfg.arm_stow) + [0.2, 0.1, 0.0, 0.2, 0.0, -0.2, 0.3]
    env.set_arm_joint_targets(target)
    env.set_gripper(True)
    for _ in range(40):
        env.wait_for_next_tick(0.05)
    state = env.get_state()
    assert state.arm_q == pytest.approx(target, abs=0.01) and np.abs(state.arm_qd).max() < 0.02
    assert state.gripper_closed_cmd and state.gripper_width < 0.002  # pads meet: the 2F-85 closes fully
    env.set_gripper(False)
    for _ in range(20):
        env.wait_for_next_tick(0.05)
    assert env.get_state().gripper_width == pytest.approx(cfg.gripper_open_width, abs=0.004)


def test_sim_tcp_matches_the_robot_layers_kinematics(stack):
    # The IK model (FR3 MJCF + configured TCP and arm mount) must agree with the simulated arm.
    stack.reset()
    env, frames = stack.env, stack.robot._arm.frames
    for q in (stack.robot_config.arm_stow, stack.robot_config.arm_ready):
        env.set_arm_joints(np.array(q))
        predicted = frames.pose_to_world(env.get_state().base_pose, frames.ik.fk(np.array(q)))
        actual = env.tcp_pose()
        assert predicted[:3, 3] == pytest.approx(actual[:3, 3], abs=0.003)
        assert predicted[:3, :3] == pytest.approx(actual[:3, :3], abs=0.01)
    assert env.arm_joint_limits == pytest.approx(frames.ik.joint_limits, abs=1e-3)  # Panda limits in sim


def test_objects_spawn_upright_on_their_fixtures(stack):
    stack.reset()
    env = stack.env
    fixtures = {f.name: f for f in named_fixtures(env)}
    spawned = {label: body for _, label, body in scene_objects(env)}
    assert set(spawned) == {"egg", "mug", "bowl", "bottled water", "pan", "cutting board"}
    for label, body in spawned.items():
        position = env.data.xpos[body]
        assert env.data.xmat[body].reshape(3, 3)[2, 2] > 0.99, f"{label} fell over"
        holder = stack.world_map.get(label).inside
        assert holder is not None and fixtures[holder].box.contains_xy(position[0], position[1])
        assert holder == ("fridge" if label == "egg" else holder) and fixtures[holder].label in ("fridge", "counter")
    assert stack.world_map.get("egg").inside == "fridge"


# -- pick ----------------------------------------------------------------------


def test_pick_success_rates_from_a_counter(stack, tmp_path_factory):
    """go_to -> detect -> align -> pick, five times per object from different
    starting poses. Each object must succeed at least 4 times out of 5."""
    starts = stack.free_starts(TRIALS, seed=11)
    rates, failures = {}, []
    for label in PICKABLE:
        successes = 0
        for trial, start in enumerate(starts):
            stack.reset()
            stack.env.teleport_base(start)
            registry = ToolRegistry(stack.robot, stack.detector, stack.world_map, AgentConfig.load())
            before = lowest_point(stack, label)
            origin = stack.object_position(label)
            contacts = stack.env.contact_ticks
            result = fetch(lambda name, **args: registry.dispatch(name, args), label)
            ok = result.success and lifted(stack, label, before) and stack.robot.get_held() is not None
            if ok:
                assert stack.env.contact_ticks == contacts, f"{label}: unexpected contact during a successful pick"
            else:
                rise = lowest_point(stack, label) - before
                held = stack.robot.get_held() is not None
                where = np.round(stack.object_position(label), 2).tolist()
                failures.append(
                    f"{label} trial {trial}: {result.error} {result.message} "
                    f"(rose {rise:.3f} m, held {held}, now at {where}, started at {np.round(origin, 2).tolist()})"
                )
            successes += ok
        rates[label] = successes
    report = {"layout": stack.layout, "trials": TRIALS, "successes": rates, "failures": failures}
    path = tmp_path_factory.getbasetemp() / f"pick_rates_layout{stack.layout}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\npick success (layout {stack.layout}): " + ", ".join(f"{k} {v}/{TRIALS}" for k, v in rates.items()))
    assert all(count >= 4 for count in rates.values()), report


def test_pick_without_align_from_far_away_is_unreachable(session):
    stack = session.stack
    mug = stack.object_position("mug")
    # A free spot 1.3-1.6 m from the mug from which the camera sees it.
    free = stack.grid.inflate(stack.robot_config.inflation_radius + 0.05)
    seen = None
    for angle in np.linspace(0.0, 2 * math.pi, 24, endpoint=False):
        for distance in (1.3, 1.45, 1.6):
            x, y = mug[0] + distance * math.cos(angle), mug[1] + distance * math.sin(angle)
            if not free.is_free(x, y):
                continue
            stack.env.teleport_base(Pose2D(x, y, math.atan2(mug[1] - y, mug[0] - x)))
            seen = session.call("detect", classes=["mug"])
            if ids(seen):
                break
        if seen is not None and ids(seen):
            break
    assert seen is not None and ids(seen) == ["mug"], "no far-away spot with a view of the mug"
    result = session.call("pick", id="mug")
    assert result.error == "unreachable" and "align" in result.message
    assert stack.object_position("mug") == pytest.approx(mug, abs=1e-3)
    assert session.call("holding").data["holding"] is None


def test_wrong_kind_calls_fail_through_the_normal_path(session):
    assert session.call("go_to", place="mug").success
    session.call("detect", classes=["mug", "cutting board", "counter"])
    counter = session.stack.world_map.get("mug").inside
    assert session.call("align", id=counter).success
    not_graspable = session.call("pick", id=counter)
    assert not_graspable.error == "no_grasp" and "no graspable object" in not_graspable.message
    assert session.call("open", id="mug").error == "unsupported"
    if "cutting board" in ids(session.call("nearby"), "visible"):
        assert session.call("pick", id="cutting board").error in ("no_grasp", "unreachable")


# -- carry and place -----------------------------------------------------------


def test_carry_a_mug_to_the_stove_and_place_it_on_a_burner(session):
    stack, env, cfg = session.stack, session.stack.env, session.stack.robot_config
    contacts = env.contact_ticks
    assert fetch(session.call, "mug").success
    assert session.call("holding").data["holding"] == {"id": "mug", "label": "mug"}
    # Held: the detector no longer reports it, and nearby says so.
    assert ids(session.call("detect", classes=["mug"])) == []
    assert session.call("nearby").message.endswith("Holding: mug.")

    assert session.call("go_to", place="stove").success
    assert env.get_state().arm_q == pytest.approx(np.array(cfg.arm_carry), abs=cfg.joint_tolerance + 0.01)
    assert env.grasped_bodies(), "the mug did not survive the drive"
    burners = ids(session.call("detect", classes=["burner"]))
    assert burners
    placed, tried = None, []
    for burner in burners:  # nearest first; a back burner may be out of reach
        aligned = session.call("align", id=burner)
        tried.append(aligned.message)
        if not aligned.success:
            continue
        placed = session.call("place", id=burner)
        tried.append(placed.message)
        if placed.success:
            break
    assert placed is not None and placed.success, tried
    target = session.registry.tools.detections.get(burner).position_world
    mug = stack.object_position("mug")
    assert np.linalg.norm(mug[:2] - target[:2]) <= 0.05, "the mug is more than 5 cm from the burner's center"
    assert env.data.xmat[body_of(stack, "mug")].reshape(3, 3)[2, 2] > 0.95  # upright
    assert session.call("holding").data["holding"] is None
    assert ids(session.call("detect", classes=["mug"])) == ["mug_1"]  # back in the scene, under a fresh id
    assert env.contact_ticks == contacts, f"unexpected contact during a successful run: {env.last_contact}"


# -- open / close --------------------------------------------------------------


def test_open_and_close_the_fridge(session):
    stack, env = session.stack, session.stack.env
    fridge = Detection("fridge", np.append(stack.world_map.get("fridge").location, 0.9))
    actuator = stack.robot._fixture_actuator
    assert actuator.get_state(fridge) == CLOSED
    assert session.call("go_to", place="fridge").success
    assert ids(session.call("detect", classes=["egg"])) == []  # behind the door
    handle = ids(session.call("detect", classes=["fridge handle"]))[0]
    assert session.call("align", id=handle).success
    contacts = env.contact_ticks

    opened = session.call("open", id=handle)
    assert opened.success and opened.data["scripted"] is True
    assert actuator.get_state(fridge) == OPEN
    assert ids(session.call("detect", classes=["egg"])) == ["egg"]
    again = session.call("open", id="fridge")
    assert again.success and "already open" in again.message

    closed = session.call("close", id="fridge")
    assert closed.success and closed.data["scripted"] is True
    assert actuator.get_state(fridge) == CLOSED
    assert ids(session.call("detect", classes=["egg"])) == []
    assert env.contact_ticks == contacts, f"the door hit the robot: {env.last_contact}"


def test_scripted_actuation_by_fixture_type(stack):
    """Every kind of handle the detector knows opens and closes its fixture."""
    stack.reset()
    env, detector, actuator = stack.env, stack.detector, stack.robot._fixture_actuator
    handles = {}
    with env.lock:
        for item in detector._items:
            if item.label.endswith("handle") and item.label not in handles:
                handles[item.label] = Detection(item.label, detector._position(item))
    assert {"fridge handle", "microwave handle", "dishwasher handle", "cabinet handle"} <= set(handles)
    for label, handle in handles.items():
        if label == "faucet":
            continue
        assert actuator.get_state(handle) == CLOSED, label
        swept = actuator.swept_xy(handle, True)
        result = actuator.set_state(handle, True)
        assert result.success and result.scripted and result.fixture, label
        assert actuator.get_state(handle) == OPEN, label
        assert swept is not None, label
        if handle.position_world[2] < 1.5:  # doors above head height sweep no floor space
            assert len(swept) > 0, label
        assert actuator.set_state(handle, False).obstacle_xy is None
        assert actuator.get_state(handle) == CLOSED, label
    assert actuator.get_state(Detection("mug", stack.object_position("mug"))) == UNKNOWN
    stack.reset()


# -- the demo --------------------------------------------------------------------


def test_demo_manip_trace(session):
    stack, env = session.stack, session.stack.env
    contacts = env.contact_ticks
    steps = []

    def run(name, args):
        result = session.registry.dispatch(name, args)
        steps.append((name, result))
        return result

    finished = manip_trace(run, stack.world_map)
    failed = "\n".join(f"{name}: {r.error} {r.message}" for name, r in steps if not r.success)
    assert finished, f"{failed}\nlast contact: {env.last_contact}"
    assert [name for name, _ in steps] == [
        "go_to", "detect", "align", "open", "detect", "align", "pick", "close",
        "go_to", "detect", "align", "place", "holding",
    ]  # fmt: skip
    counter = next(f for f in named_fixtures(env) if f.name == "counter_1")
    egg = stack.object_position("egg")
    assert counter.box.contains_xy(egg[0], egg[1]) and abs(egg[2] - counter.box.top) < 0.06
    assert stack.robot._fixture_actuator.get_state(Detection("fridge", np.append(stack.world_map.get("fridge").location, 0.9))) == CLOSED
    assert env.contact_ticks == contacts, f"unexpected contact during the demo: {env.last_contact}"
