"""Register a lab run's final skill: copy it into skills/<goal skill>/<version>/ with a manifest and the run's report.

    python -m chefbot.autoresearch.register runs/labs/<run> --version run1_v12 [--lab labs/<name>.yaml] [--from skills/final]

The manifest's robot and model access come from the lab's config; the models listed are the ones the skill's code
actually names (Gemini models in robot.llm calls; vision models in detect/segment calls).
(agentic-skills-exploration's autoresearch/register.py.)
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import yaml

from chefbot.autoresearch import config as C
from chefbot import setups, skills, suites


def models_in(code: str) -> dict:
    out = {}
    consts = dict(re.findall(r"^(\w+)\s*=\s*['\"]([\w.\-]+)['\"]", code, re.M))  # llm(MODEL, ...) with MODEL = "gemini-..."
    named = re.findall(r"llm\(\s*['\"]([\w.\-]+)['\"]", code) + [consts[n] for n in re.findall(r"llm\(\s*(\w+)\s*,", code) if n in consts]
    if re.search(r"\.llm\(", code) and not named:
        print("warning: the skill calls robot.llm with a model chosen at run time; add it to the manifest's models by hand")
    for m in sorted(set(named)):
        out[m] = {"runs": "api", "provider": "google", "key": "GEMINI_API_KEY"}
    vision = set(re.findall(r"model\s*=\s*['\"](owlv2|grounding-dino|sam2)['\"]", code))
    if re.search(r"\.detect\(", code) and not vision & {"owlv2", "grounding-dino"}:
        vision.add("owlv2")  # detect's default
    if re.search(r"\.segment\(", code):
        vision.add("sam2")
    for m in sorted(vision):
        out[m] = {"runs": "vision_models"}
    return out


def envs_of(cfg: C.LabConfig) -> list[str]:
    """Where a lab's skill could run: the environments of its dev suite's setups."""
    return sorted({setups.load(item.setup).environment for item in suites.load(cfg.suite("dev")).items})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the lab run's directory (runs/labs/<run>)")
    ap.add_argument("--version", required=True)
    ap.add_argument("--lab", help="its lab config (default: read from the run's private/lab.yaml)")
    ap.add_argument("--from", dest="src", default="skills/final", help="the skill in the workspace")
    a = ap.parse_args()
    cfg = C.load(a.lab or str(a.run / "private" / "lab.yaml"))
    if not a.lab and (a.run / "private" / "lab_name").exists():
        cfg.name = (a.run / "private" / "lab_name").read_text().strip()  # the copy is private/lab.yaml: its own name
    ws = a.run / "workspace"
    src = ws / a.src
    dst = skills.ROOT / cfg.goal["skill"] / a.version
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if (ws / "REPORT.md").exists():
        shutil.copy(ws / "REPORT.md", dst / "REPORT.md")
    code = "\n".join(p.read_text() for p in dst.rglob("*.py"))
    manifest = {"name": cfg.goal["skill"], "version": a.version, "entry": "main.py:run", "interface": cfg.interface,
                "envs": envs_of(cfg), "sandboxed": cfg.interface != "robot-api-v1", "robot": cfg.robot,
                **({"arm_only": True} if cfg.arm_only else {}), **({"args": cfg.raw_args} if cfg.raw_args else {}),
                "speed": "normal", "models": models_in(code), "assumes": "the start rule of its lab's task",
                "provenance": {"lab": cfg.name, "run": a.run.name, "suite": cfg.suite("dev").split("/")[0], "test": "not tested", "report": "REPORT.md"}}  # fmt: skip
    (dst / "skill.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"registered {dst.relative_to(skills.ROOT)}; test it: python -m chefbot.evals.evaluate --skill {cfg.goal['skill']}/{a.version} "
          f"--suite {cfg.suite('test')}")  # fmt: skip


if __name__ == "__main__":
    main()
