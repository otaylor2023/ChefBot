"""Every skill's manifest parses and loads (a stray ': ' in a note broke run_real twice on 2026-11-04)."""

import pytest
import yaml

from chefbot import skills
from chefbot.skills import ROOT


@pytest.mark.parametrize("path", sorted(ROOT.glob("*/*/skill.yaml")), ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_the_manifest_parses_and_loads(path):
    raw = yaml.safe_load(path.read_text())
    assert isinstance(raw, dict) and raw.get("entry"), path
    s = skills.load(f"{path.parent.parent.name}/{path.parent.name}")
    s.access()
