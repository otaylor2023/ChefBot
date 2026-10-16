"""Layer boundaries from NAV_SPEC.md section 2, enforced on the import graph."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent
REPO = PACKAGE.parent
SIM_AND_HARDWARE = ("robosuite", "robocasa", "isaaclab", "isaacsim", "omni", "chefbot.envs")
KINEMATICS_ONLY = ("mujoco", "mink")


def modules_under(layer: str) -> list[Path]:
    return sorted((PACKAGE / layer).rglob("*.py"))


def imports_of(path: Path) -> set[str]:
    """Every module a file imports, as absolute dotted names."""
    package = ".".join(path.relative_to(REPO).with_suffix("").parts[:-1])
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                base = ".".join(parent + ([base] if base else []))
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def violations(layer: str, forbidden: tuple[str, ...], allow_under: str | tuple[str, ...] = ()) -> list[str]:
    out = []
    for path in modules_under(layer):
        relative = path.relative_to(PACKAGE).as_posix()
        if allow_under and relative.startswith(allow_under):
            continue
        for module in imports_of(path):
            if any(module == name or module.startswith(name + ".") for name in forbidden):
                out.append(f"{relative} imports {module}")
    return sorted(out)


def test_layers_exist():
    for layer in ("agent", "robot", "interface", "envs"):
        assert modules_under(layer), layer


@pytest.mark.parametrize("layer", ["agent", "robot", "interface", "grasping"])
def test_no_simulator_or_hardware_imports(layer):
    assert violations(layer, SIM_AND_HARDWARE) == []


def test_mujoco_and_mink_only_in_the_robot_kinematics_code():
    assert violations("agent", KINEMATICS_ONLY) == []
    assert violations("interface", KINEMATICS_ONLY) == []
    # The one IK module (MANIP_SPEC §5.2) and the reachability checker built on it.
    assert violations("robot", KINEMATICS_ONLY, allow_under=("robot/kinematics/", "robot/arm/ik.py")) == []


def test_agent_never_sees_grasps_joints_or_tcp_frames():
    # MANIP_SPEC §10: the agent deals in ids and results only.
    hidden = tuple(f"chefbot.interface.types.{name}" for name in ("Grasp", "Placement", "EnvState"))
    assert violations("agent", hidden + ("chefbot.interface.proposers", "chefbot.interface.fixtures")) == []


def test_agent_knows_only_the_interface():
    assert violations("agent", ("chefbot.robot",)) == []


def test_interface_depends_on_no_other_layer():
    assert violations("interface", ("chefbot.agent", "chefbot.robot")) == []


def test_robot_does_not_import_the_agent():
    assert violations("robot", ("chefbot.agent",)) == []


def test_ground_truth_never_reaches_the_agents_context():
    # AGENT_SPEC §2: Observer output goes to the StateChecker and the run log only.
    for name in ("agent/loop/context.py", "agent/loop/perception.py"):
        imported = imports_of(PACKAGE / name)
        assert not any(m.startswith(("chefbot.agent.check", "chefbot.interface.observer")) for m in imported), name


def test_only_the_loop_and_the_checker_talk_to_an_llm():
    users = sorted(
        path.relative_to(PACKAGE).as_posix()
        for path in modules_under("agent")
        if any(m == "chefbot.agent.loop.llm" or m.startswith("chefbot.agent.loop.llm.") for m in imports_of(path))
    )
    assert all(u.startswith(("agent/loop/", "agent/check/")) for u in users), users


def test_grasping_depends_only_on_the_interface():
    # Grasp planning from perceived geometry, shared by every env's grasp proposer.
    assert violations("grasping", ("chefbot.agent", "chefbot.robot") + KINEMATICS_ONLY) == []


def test_envs_do_not_import_the_robot_layer_or_the_agent():
    # serve.py is the composition root that wires Robot(Env) to the server.
    assert violations("envs", ("chefbot.robot", "chefbot.agent"), allow_under="envs/robocasa/serve.py") == []


def run_pytest_without(blocked: list[str], targets: list[str]) -> subprocess.CompletedProcess:
    """Run pytest in a fresh interpreter where importing `blocked` modules fails."""
    code = (
        "import sys\n"
        f"for name in {blocked!r}: sys.modules[name] = None\n"
        "import pytest\n"
        f"sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', *{targets!r}]))\n"
    )
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)


def test_agent_suite_passes_without_any_simulator():
    result = run_pytest_without(
        ["robosuite", "robocasa", "mujoco", "mink"], ["chefbot/tests/agent"]
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def test_robot_and_transport_suites_pass_without_robosuite_or_robocasa():
    result = run_pytest_without(
        ["robosuite", "robocasa"], ["chefbot/tests/robot", "chefbot/tests/transport", "chefbot/tests/grasping"]
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
