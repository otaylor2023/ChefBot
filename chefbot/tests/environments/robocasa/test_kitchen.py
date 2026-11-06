"""RoboCasa kitchen state: burners on the map and through the robot, relations, cooking over skipped time, safety."""

import mujoco
import numpy as np
import pytest

from chefbot.agent.config import AgentConfig
from chefbot.agent.tools.registry import ToolRegistry
from chefbot.environments.robocasa.appliances import burner_on, burner_position, burners, set_burner
from chefbot.environments.robocasa.detector import scene_objects
from chefbot.environments.robocasa.kitchen_state import RoboCasaClock, RoboCasaKitchen
from chefbot.environments.robocasa.objects import object_points
from chefbot.environments.symbolic.world import Timing

pytestmark = pytest.mark.robocasa


def front_burner(env):
    return next(b for b in burners(env) if b.location.startswith("front"))


def put(env, label, xyz):
    body = next(b for _, l, b in scene_objects(env) if l == label)
    joint = env.model.body_jntadr[body]
    adr, dof = env.model.jnt_qposadr[joint], env.model.jnt_dofadr[joint]
    with env.lock:
        env.data.qpos[adr : adr + 3] = xyz
        env.data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(env.model, env.data)


def settle(env, ticks=25):
    for _ in range(ticks):
        env.wait_for_next_tick(0.05)


def test_burners_are_on_the_map_and_turn_on_and_off_through_the_robot(stack):
    stack.reset()
    env = stack.env
    names = {e.name for e in stack.world_map.list()}
    assert {b.name for b in burners(env)} <= names  # stable names for the agent
    burner = front_burner(env)
    registry = ToolRegistry(stack.robot, stack.detector, stack.world_map, AgentConfig.load())
    assert "turn_on" in {s["function"]["name"] for s in registry.schemas}  # the robot has APPLIANCES now
    assert registry.dispatch("go_to", {"place": burner.name}).success
    seen = registry.dispatch("detect", {"classes": ["burner"]})
    assert {"id": burner.name, "label": "burner", "state": "off"} in seen.data["detections"]
    assert registry.dispatch("align", {"id": burner.name}).success
    on = registry.dispatch("turn_on", {"id": burner.name})
    assert on.success and on.data.get("scripted") and burner_on(env, burner)
    again = registry.dispatch("detect", {"classes": ["burner"]})
    assert {"id": burner.name, "label": "burner", "state": "on"} in again.data["detections"]
    assert registry.dispatch("turn_off", {"id": burner.name}).success and not burner_on(env, burner)


def test_food_in_a_pan_on_a_lit_burner_cooks_while_idle_time_is_skipped(stack):
    stack.reset()
    env = stack.env
    timing = Timing.load()
    kitchen = RoboCasaKitchen(env, stack.robot._fixture_actuator, timing)
    clock = RoboCasaClock(kitchen)
    burner = front_burner(env)
    put(env, "pan", burner_position(env, burner) + [0, 0, 0.03])
    settle(env)
    pan = next(b for _, l, b in scene_objects(env) if l == "pan")
    put(env, "egg", env.data.xpos[pan] + [0, 0, 0.06])
    settle(env)
    text = kitchen.describe()
    assert f"pan: on {burner.name}" in text and "egg: inside pan" in text
    with env.lock:
        set_burner(env, burner, True, np.random.RandomState(0))
    kitchen.update(force=True)
    sim_before = float(env.data.time)
    set_at = dict(timing.cooking["egg"])["set"]
    clock.sleep(set_at + 10)  # skipped: no physics
    assert float(env.data.time) == pytest.approx(sim_before)
    text = kitchen.describe()
    assert "egg: cook state = set" in text and f"{burner.name}: on (for " in text
    with env.lock:
        set_burner(env, burner, False, np.random.RandomState(0))


def test_an_empty_lit_burner_is_turned_off(stack):
    stack.reset()
    env = stack.env
    kitchen = RoboCasaKitchen(env, stack.robot._fixture_actuator)
    clock = RoboCasaClock(kitchen)
    burner = front_burner(env)
    with env.lock:
        set_burner(env, burner, True, np.random.RandomState(0))
    kitchen.update(force=True)
    clock.sleep(kitchen.timing.empty_burner_s + 2)
    assert not burner_on(env, burner)
    [(message, _)] = kitchen.safety(clock.now())
    assert "nothing on it" in message


def test_food_rides_in_a_held_pan_and_pours_into_a_bowl(stack):
    stack.reset()
    env = stack.env
    kitchen = RoboCasaKitchen(env, stack.robot._fixture_actuator)  # pans and utensils lock in the grip
    pan = next(b for _, l, b in scene_objects(env) if l == "pan")
    points = object_points(env, pan)
    floor = points[points[:, 2] < points[:, 2].min() + 0.01]
    put(env, "egg", (floor.min(axis=0) + floor.max(axis=0)) / 2 + [0, 0, 0.04])  # in the pan's bowl
    settle(env)
    assert "egg: inside pan" in kitchen.describe()
    registry = ToolRegistry(stack.robot, stack.detector, stack.world_map, AgentConfig.load())

    def call(name, **args):
        result = registry.dispatch(name, args)
        assert result.success, result.message
        return result

    call("go_to", place="pan")
    call("detect", classes=["pan"])
    call("align", id="pan")
    call("pick", id="pan")
    call("go_to", place="bowl")
    egg = env.data.xpos[next(b for _, l, b in scene_objects(env) if l == "egg")]
    assert np.linalg.norm(egg[:2] - env.data.xpos[pan][:2]) < 0.15, "the egg fell out of the pan on the way"
    call("detect", classes=["bowl"])
    call("align", id="bowl")
    call("pour", id="bowl")
    settle(env)
    assert "egg: inside bowl" in kitchen.describe()
    assert registry.dispatch("holding", {}).data["holding"]["label"] == "pan"  # still holding it, level again
