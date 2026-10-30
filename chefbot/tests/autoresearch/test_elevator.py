"""The elevator family's checks: the example building passes them, and deliberately broken buildings fail the right
ones (so the checks catch what they claim to)."""

import json
import shutil
import sys
from pathlib import Path

import pytest

from chefbot.autoresearch.generator_lab import family
from chefbot.generators.elevator import checks

EXAMPLE = family.GENERATORS / "elevator" / "example"


@pytest.fixture(scope="module")
def fam():
    return family.load("elevator")


def build(tmp: Path, overrides=None) -> tuple[Path, dict]:
    sys.path.insert(0, str(EXAMPLE))
    try:
        import generate
    finally:
        sys.path.pop(0)
    out = tmp / "b"
    out.mkdir(parents=True)
    params = generate.generate(0, str(out), overrides)
    (out / "params.json").write_text(json.dumps(params))
    return out, params


def edit_dynamics(folder: Path, old: str, new: str) -> None:
    p = folder / "dynamics.py"
    s = p.read_text()
    assert old in s, old
    p.write_text(s.replace(old, new))


def test_the_example_passes_every_check(tmp_path, fam):
    out, params = build(tmp_path)
    r = checks.run(out, fam, params)
    assert r["ok"], {k: v.get("why") for k, v in r["checks"].items() if not v.get("ok")}
    m = r["measured"]
    assert m["realtime_x"] > 1 and abs(m["level_error_m"]) < 0.007 and m["max_speed_mps"] <= params["run"]["rated_speed_mps"] * 1.1


def test_a_wide_sill_gap_fails_the_trip(tmp_path, fam):
    out, params = build(tmp_path, {"sill_gap_m": 0.06})
    params["build"]["sill_gap_m"] = 0.06
    r = checks.run(out, fam, params)
    assert "trip" in r["failed"] and "sill gap" in r["checks"]["trip"]["why"]


def test_doors_that_ignore_an_obstruction_fail(tmp_path, fam):
    out, params = build(tmp_path)
    edit_dynamics(out, "            if blocked or straining or held:", "            if False:")
    edit_dynamics(out, "                    elif self.timer > 0.3:", "                    elif False:")
    r = checks.run(out, fam, params)
    assert "obstruction" in r["failed"], r["checks"]["obstruction"]


def test_a_car_faster_than_declared_fails(tmp_path, fam):
    out, params = build(tmp_path)
    params["run"]["rated_speed_mps"] = round(params["run"]["rated_speed_mps"] / 2, 3)  # declared, not what it does
    r = checks.run(out, fam, params)
    assert "trip" in r["failed"] and "speed" in r["checks"]["trip"]["why"]


def test_a_landing_door_open_without_the_car_is_unsafe(tmp_path, fam):
    out, params = build(tmp_path)
    edit_dynamics(out, "            frac = self.door if floor == here and self.phase != \"moving\" else 0.0",
                  "            frac = 1.0 if floor == self.floor_ids[0] else (self.door if floor == here and self.phase != \"moving\" else 0.0)")
    r = checks.run(out, fam, params)
    assert not r["ok"] and "no car there" in json.dumps(r["checks"])
