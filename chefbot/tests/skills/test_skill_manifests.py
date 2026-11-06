"""Skill manifests (skills/<name>/<version>/skill.yaml): what the loader accepts, refuses and lists."""

import shutil
from pathlib import Path

import pytest
import yaml

from chefbot import skills

ROBOT = [f"{n}/robotapi_v1" for n in ("align", "close", "go_to", "lift_out", "lower_into", "open", "pick_up", "place", "pour",
                                     "release", "squeeze", "stir", "turn_off", "turn_on")]  # the agent's, robot-api-v1  # fmt: skip
LAB = ["bringup/v1", "pick_up/astra_v19", "pick_up/chefbot_v30", "pick_up/chefbot_v30_fr3", "pick_up/gemini_v1",
              "pick_up/run1_v32", "tongs_lift/mujoco_tongs1_v34", "tongs_lift/mujoco_tongs2_v17", "tongs_lift/mujoco_tongs3_v59"]  # fmt: skip
REGISTERED = sorted(LAB + ROBOT)
IMPORTED = {"pick_up/astra_v19", "pick_up/run1_v32"}  # wholebody-v2, from agentic-skills-exploration


def test_every_registered_skill_loads():
    assert skills.names() == REGISTERED
    for path in REGISTERED:
        s = skills.load(path)
        assert s.sandboxed == (path in LAB)  # the lab's run as their own process; the agent's in its process
        assert s.runnable == (path not in IMPORTED)


def test_the_catalog_lists_by_interface():
    assert [s.path for s in skills.catalog(interface="chefbot-v1")] == [p for p in LAB if p not in IMPORTED]
    assert [s.path for s in skills.catalog(interface="robot-api-v1")] == ROBOT
    assert len(skills.catalog()) == 23


def test_the_catalog_lists_by_env():
    for s in skills.catalog():
        for env in s.envs:
            assert s in skills.catalog(env=env)
    assert all(env in skills.ENVS for s in skills.catalog() for env in s.envs)


def test_each_skill_says_where_it_could_run():
    def at(env):
        return [s.path for s in skills.catalog(env=env)]

    assert at("symbolic") == ROBOT
    assert at("robocasa") == sorted(["bringup/v1", "pick_up/chefbot_v30"] + ROBOT)
    assert at("mujoco_kitchen") == ["bringup/v1", "tongs_lift/mujoco_tongs1_v34", "tongs_lift/mujoco_tongs2_v17",
                                    "tongs_lift/mujoco_tongs3_v59"]  # fmt: skip
    assert at("real") == ["bringup/v1", "pick_up/chefbot_v30_fr3", "pick_up/gemini_v1"]
    for path in IMPORTED:
        assert skills.load(path).envs == ()  # runs nowhere here


def test_a_skills_envs_match_its_labs_suite():
    from chefbot.autoresearch import config as C
    from chefbot.autoresearch.register import envs_of

    assert envs_of(C.load("chefbot/labs/robocasa_pick.yaml")) == ["robocasa"]
    assert envs_of(C.load("chefbot/labs/mujoco_tongs.yaml")) == ["mujoco_kitchen"]
    assert skills.load("pick_up/chefbot_v30").envs == ("robocasa",)
    assert all(skills.load(p).envs == ("mujoco_kitchen",) for p in LAB if p.startswith("tongs_lift/"))


def test_the_catalog_refuses_what_it_does_not_know():
    with pytest.raises(ValueError, match="no env 'moon'"):
        skills.catalog(env="moon")
    with pytest.raises(ValueError, match="no interface 'wholebody-v2'"):
        skills.catalog(interface="wholebody-v2")  # imported, but no session speaks it


def _copy(tmp_path: Path, edit, family="bringup", version="v1") -> Path:
    d = tmp_path / family / version
    shutil.copytree(skills.ROOT / "bringup" / "v1", d, ignore=shutil.ignore_patterns("__pycache__"))
    m = yaml.safe_load((d / "skill.yaml").read_text())
    edit(m)
    (d / "skill.yaml").write_text(yaml.safe_dump(m))
    return d


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda m: m.update(colour="blue"), "unknown keys"),
        (lambda m: m.update(name="pick_up"), "is not its folder's"),
        (lambda m: m.update(version="v2"), "is not its folder's"),
        (lambda m: m.update(interface="wholebody-v3"), "interface must be one of"),
        (lambda m: m.update(sandboxed=False), "always sandboxed"),
        (lambda m: m.update(interface="robot-api-v1", sandboxed=True), "cannot be sandboxed"),
        (lambda m: m.update(sandboxed="yes"), "true or false"),
        (lambda m: m.pop("sandboxed"), "no sandboxed"),
        (lambda m: m.pop("envs"), "envs must be some of"),
        (lambda m: m.update(interface="wholebody-v2"), "runs nowhere here"),
        (lambda m: m.update(envs=["moon"]), "envs must be some of"),
        (lambda m: m.update(envs=[]), "envs must be some of"),
        (lambda m: m.update(envs="real"), "envs must be some of"),
        (lambda m: m.update(entry="skill.py:run"), "is not file:function"),
        (lambda m: m.update(entry="main.py"), "is not file:function"),
        (lambda m: m["args"]["lift_m"].update(type="float"), "must have a type"),
    ],
)
def test_a_broken_manifest_is_refused(tmp_path, edit, message):
    with pytest.raises(ValueError, match=message):
        skills.load(str(_copy(tmp_path, edit)))


def test_a_robot_api_skill_is_never_sandboxed(tmp_path):
    s = skills.load(str(_copy(tmp_path, lambda m: m.update(interface="robot-api-v1", sandboxed=False, description="Lift."))))
    assert not s.sandboxed and s.runnable


def test_a_manifest_may_say_where_it_could_run(tmp_path):
    s = skills.load(str(_copy(tmp_path, lambda m: m.update(envs=["robocasa", "real"]))))
    assert s.envs == ("robocasa", "real") and s.sandboxed


def test_what_the_runtime_gets_is_unchanged_by_the_new_fields(tmp_path):
    before = skills.load("bringup/v1")
    after = skills.load(str(_copy(tmp_path, lambda m: m.update(envs=["real"], sandboxed=True))))
    assert {k: v for k, v in after.spec().items() if k not in ("dir", "path")} == {
        k: v for k, v in before.spec().items() if k not in ("dir", "path")
    }
