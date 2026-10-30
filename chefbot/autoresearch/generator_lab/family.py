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
    return module(fam, "checks")


def module(fam: dict, name: str):
    """One of the family's modules: checks (run), render (render_all)."""
    return importlib.import_module(f"chefbot.generators.{fam['name']}.{name}")


def judging(fam: dict) -> dict:
    """What the judge scores and the names of the images it reads (family.yaml `judge`)."""
    j = fam.get("judge") or {}
    return {"scores": list(j.get("scores") or ["shape", "tips", "mechanism", "materials", "proportions", "overall"]),
            "motion": j.get("motion_image", "asset_squeeze.jpg"), "refs": j.get("refs_image", "real_tongs.jpg")}  # fmt: skip
