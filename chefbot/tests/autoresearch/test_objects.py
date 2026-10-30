"""The object-creation lab without a judge: the example tongs generator, run sandboxed, passes the checks that a crude
generator can; the checks catch broken contracts; the generator's sandbox blocks the repo and the network."""

import json
import shutil
import textwrap
from pathlib import Path

import pytest

from chefbot.autoresearch.objects import checks, generate, pipeline, scene
from chefbot.autoresearch.objects import lab as OL

EXAMPLE = Path(__file__).resolve().parents[2] / "autoresearch" / "templates" / "example_tongs"


@pytest.fixture(scope="module")
def fam():
    return checks.load_family("tongs")


def test_the_example_builds_and_passes_the_physics_checks(tmp_path, fam):
    rec = pipeline.build_one(str(EXAMPLE), 3, str(tmp_path / "s3"), fam, video=False)
    assert rec["ok"], rec.get("why")
    m = rec["measured"]
    assert 0.2 < m["length_m"] < 0.4 and 0 < m["closing_force_N"] < 50
    assert rec.get("rendered") and (tmp_path / "s3" / "lab" / "sheet.jpg").exists()
    c = json.loads((tmp_path / "s3" / "lab" / "checks.json").read_text())["checks"]
    assert not c["squeeze"]["passes_through"] and c["release"]["ok"] and c["counter"]["lies_on_side"]


def test_a_model_without_both_arms_fails_cleanly(tmp_path, fam):
    """arm_a folded into a hingeless body and arm_b renamed: the checks report it, they don't crash."""
    out = tmp_path / "a"
    assert generate.run(EXAMPLE, 3, out, 60, 20)["ok"]
    xml = (out / "model.xml").read_text()
    start = xml.index('<body name="arm_a"')
    body_end = xml.index("</body>", start) + len("</body>")
    inner = xml[xml.index(">", start) + 1 : xml.index("</body>", start)]
    inner = inner[inner.index("/>", inner.index("<joint")) + 2 :]
    xml = xml[:start] + '<body name="arm_a" pos="0 0 0">' + inner + "</body>" + xml[body_end:]
    xml = xml.replace('<body name="arm_b"', '<body name="arm_b_inner"', 1)
    (out / "model.xml").write_text(xml)
    with pytest.raises(Exception):
        scene.natural_frame(out)  # arm_b is gone: a contract error, not a crash
    r = checks.run(out, fam)
    assert not r["ok"]


def test_contract_errors_are_reported(tmp_path, fam):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "model.xml").write_text('<mujoco><compiler angle="radian"/><worldbody><body name="thing"><geom size="0.01"/></body></worldbody></mujoco>')
    r = checks.run(d, fam, params={})
    assert not r["ok"] and "object" in r["checks"]["structure"]["why"]


def test_the_generators_sandbox_blocks_the_repo_and_the_network(tmp_path):
    gen = tmp_path / "gen"
    gen.mkdir()
    repo_file = Path(__file__).resolve()
    (gen / "generate.py").write_text(textwrap.dedent(f"""
        def generate(seed, out_dir):
            got = []
            for attempt in (lambda: open({str(repo_file)!r}).read(), lambda: __import__("socket").create_connection(("example.com", 80), 2)):
                try:
                    attempt()
                    got.append("ALLOWED")
                except Exception as e:
                    got.append(type(e).__name__)
            return {{"got": got}}
    """))
    r = generate.run(gen, 0, tmp_path / "out", 60, 20)
    assert r["ok"] and "ALLOWED" not in r["params"]["got"], r


def test_the_lab_config_and_brief_fill_in(tmp_path):
    cfg = OL.load("tongs_gen")
    assert OL.kind("tongs_gen") == "object" and cfg.family == "tongs"
    text = OL.brief(cfg, 3, 30, "bwrap", True)
    assert "{" not in text.replace("{seed}", ""), [l for l in text.splitlines() if "{" in l]
    OL.setup_workspace(cfg, tmp_path / "ws", text)
    ws = tmp_path / "ws"
    for f in ("AGENTS.md", "lab", "docs/generator_contract.md", "docs/checks.md", "docs/judge_rubric.md", "generators/example/generate.py"):
        assert (ws / f).exists(), f
    assert "{" not in (ws / "docs" / "generator_contract.md").read_text().split("```")[0]


def test_the_robocasa_export_keeps_what_robosuite_would_drop(tmp_path):
    """Collision geoms end up in group 0 and visual ones in group 1 (robosuite keeps no others), single sizes are
    padded for RoboCasa's scaler, a reg_bbox region bounds it, and the folder is marked for ChefBot's spawner."""
    import xml.etree.ElementTree as ET

    from chefbot.autoresearch.objects import robocasa as RC
    from chefbot.envs.robocasa.spawn import GENERATED_MARK

    asset = tmp_path / "a"
    assert generate.run(EXAMPLE, 3, asset, 60, 20)["ok"]
    xml = (asset / "model.xml").read_text().replace('group="0"', 'group="3"').replace('size="0.004"', 'size="0.004"', 1)
    (asset / "model.xml").write_text(xml)
    out = RC.export(asset, tmp_path / "rc")
    root = ET.parse(out / "model.xml").getroot()
    obj = root.find("worldbody/body/body")
    assert obj.get("name") == "object" and (out / GENERATED_MARK).exists()
    groups = {(g.get("group"), g.get("contype") == "0" and g.get("conaffinity") == "0") for g in obj.iter("geom") if g.get("name") != "reg_bbox"}
    assert groups <= {("0", False), ("1", True)}
    assert all(len(g.get("size", "1 1").split()) >= 2 for g in obj.iter("geom"))
    assert obj.find("geom[@name='reg_bbox']") is not None
