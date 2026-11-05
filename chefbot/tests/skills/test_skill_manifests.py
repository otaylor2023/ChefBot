"""Skill manifests (skills/<name>/<version>/skill.yaml): what the loader accepts, refuses and lists."""

import shutil
from pathlib import Path

import pytest
import yaml

from chefbot import skills

REGISTERED = ["bringup/v1", "pick_up/astra_v19", "pick_up/chefbot_v30", "pick_up/chefbot_v30_fr3", "pick_up/gemini_v1",
              "pick_up/run1_v32", "tongs_lift/mujoco_tongs1_v34", "tongs_lift/mujoco_tongs2_v17", "tongs_lift/mujoco_tongs3_v59"]  # fmt: skip
IMPORTED = {"pick_up/astra_v19", "pick_up/run1_v32"}  # wholebody-v2, from agentic-skills-exploration


def test_every_registered_skill_loads():
    assert skills.names() == REGISTERED
    for path in REGISTERED:
        s = skills.load(path)
        assert s.sandboxed  # all of today's run as their own process
        assert s.runnable == (path not in IMPORTED)


def test_the_catalog_lists_by_interface():
    assert [s.path for s in skills.catalog(interface="chefbot-v1")] == [p for p in REGISTERED if p not in IMPORTED]
    assert skills.catalog(interface="robot-api-v1") == []  # the agent's tools become these in step 5
    assert len(skills.catalog()) == 9


def test_the_catalog_lists_by_env():
    for s in skills.catalog():
        for env in s.envs:
            assert s in skills.catalog(env=env)
    assert all(env in skills.ENVS for s in skills.catalog() for env in s.envs)


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
    s = skills.load(str(_copy(tmp_path, lambda m: m.update(interface="robot-api-v1"))))
    assert not s.sandboxed and s.runnable
    s = skills.load(str(_copy(tmp_path / "b", lambda m: m.update(interface="robot-api-v1", sandboxed=False))))
    assert not s.sandboxed


def test_a_manifest_may_say_where_it_could_run(tmp_path):
    s = skills.load(str(_copy(tmp_path, lambda m: m.update(envs=["robocasa", "real"], sandboxed=True))))
    assert s.envs == ("robocasa", "real") and s.sandboxed


def test_what_the_runtime_gets_is_unchanged_by_the_new_fields(tmp_path):
    before = skills.load("bringup/v1")
    after = skills.load(str(_copy(tmp_path, lambda m: m.update(envs=["real"], sandboxed=True))))
    assert {k: v for k, v in after.spec().items() if k not in ("dir", "path")} == {
        k: v for k, v in before.spec().items() if k not in ("dir", "path")
    }
