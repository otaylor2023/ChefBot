"""A generator family: chefbot/generators/<name>/ holds family.yaml (limits and check thresholds), the contract, the
judge's rubrics, the example generator (example/), and checks.py (the family's checks: run(asset_dir, fam, params))."""

from __future__ import annotations

import importlib
from pathlib import Path

import yaml

GENERATORS = Path(__file__).resolve().parents[2] / "generators"


def load(name: str) -> dict:
    d = GENERATORS / name
    fam = yaml.safe_load((d / "family.yaml").read_text())
    fam["dir"] = str(d)
    return fam


def checks(fam: dict):
    """The family's checks module."""
    return importlib.import_module(f"chefbot.generators.{fam['name']}.checks")
