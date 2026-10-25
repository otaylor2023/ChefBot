"""How close each object in an explored map is to where it really is.

    python -m chefbot.explore.accuracy runs/<run> [runs/<run> ...] [--rebuild-truth] [--markdown]

The scorer (score.py) only asks whether an object is within a tolerance. This
measures the errors: for each true object, the explored object of the same
label nearest to it, and

  xy, z, 3D error   against the true map's position: the object's body origin
  ... vs centre     against the middle of the object's geometry, which is what
                    a detector sees (a pan's origin sits near its handle)
  reference offset  origin to centre: the part of an error that is the true
                    map's choice of reference point, not the detector
  fixture           whether it was put in or on the right fixture

xy comes from explored_map.yaml and true_map.yaml. Heights come from the 3D
files a run writes (explored_positions.yaml, true_positions.yaml); for older
runs the explored heights are read from the detector's frame logs (vision/)
and, with --rebuild-truth, the true positions from the same scene rebuilt and
reset (placement is deterministic; an object whose rebuilt spot differs from
the true map's is left without a height). Only this analysis reads the truth,
after the run.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from chefbot.agent.world.map import label_of

MATCH_RADIUS = 0.5  # m: an explored object farther than this from the true one is not a sighting of it
BANDS = (0.05, 0.10, 0.15)  # m: report the share of objects within each of these (xy)
_SAME_SIGHTING = 0.05  # m: a logged detection this close (xy) to an explored object is its sighting
_SAME_TRUTH = 0.01  # m: a rebuilt object this close (xy) to the true map's is the same placement


def _label(name: str) -> str:
    return label_of(name).replace("_", " ")


@dataclass
class Run:
    name: str
    detector: str
    layout: int | None
    explored: dict  # name -> {"location": [x, y], "in": fixture}
    truth: dict
    explored_3d: dict = field(default_factory=dict)  # name -> [x, y, z]
    truth_3d: dict = field(default_factory=dict)  # name -> {"origin": [...], "center": [...]}


def match(explored: dict, truth: dict, radius: float = MATCH_RADIUS) -> dict[str, str | None]:
    """{true name: explored name or None}: same label within `radius` (xy), nearest first, each used once."""
    pairs = sorted(
        (d, tn, en)
        for tn, t in truth.items() for en, e in explored.items()
        if _label(en) == _label(tn) and (d := math.dist(e["location"], t["location"])) <= radius
    )  # fmt: skip
    out: dict[str, str | None] = {tn: None for tn in truth}
    used = set()
    for _, tn, en in pairs:
        if out[tn] is None and en not in used:
            out[tn] = en
            used.add(en)
    return out


def object_errors(run: Run) -> list[dict]:
    """One row per true object (found_as None when missing)."""
    rows = []
    for tn, en in match(run.explored, run.truth).items():
        t = run.truth[tn]
        row: dict = {"object": tn, "found_as": en, "true_in": t.get("in")}
        if en is not None:
            e = run.explored[en]
            row["xy"] = math.dist(e["location"], t["location"])
            row["in_ok"] = e.get("in") == t.get("in")
            row["in"] = e.get("in")
            seen, true = run.explored_3d.get(en), run.truth_3d.get(tn)
            if true is not None:
                origin, center = np.array(true["origin"]), np.array(true["center"])
                row["offset_xy"] = float(np.linalg.norm(origin[:2] - center[:2]))
                row["offset_z"] = float(abs(origin[2] - center[2]))
                row["xy_center"] = float(np.linalg.norm(np.array(e["location"]) - center[:2]))
                if seen is not None:
                    seen = np.array(seen)
                    row["z"] = float(abs(seen[2] - origin[2]))
                    row["d3"] = float(np.linalg.norm(seen - origin))
                    row["z_center"] = float(abs(seen[2] - center[2]))
                    row["d3_center"] = float(np.linalg.norm(seen - center))
        rows.append(row)
    return rows


def summarize(rows: list[dict], key: str = "xy") -> dict:
    """Matched count and the error distribution of `key` (meters) over the rows that have it."""
    values = np.array([r[key] for r in rows if key in r], dtype=float)
    out = {"objects": len(rows), "matched": sum(r.get("found_as") is not None for r in rows), "n": len(values)}
    if len(values):
        out.update(mean=float(values.mean()), median=float(np.median(values)), p90=float(np.percentile(values, 90)),
                   max=float(values.max()))  # fmt: skip
        out.update({f"within_{round(b * 100)}cm": float((values <= b).mean()) for b in BANDS})
    if any("in_ok" in r for r in rows):
        out["fixture_ok"] = float(np.mean([r["in_ok"] for r in rows if "in_ok" in r]))
    return out


# -- loading runs ----------------------------------------------------------------------


def load_run(run_dir: str | Path, rebuild_truth: bool = False) -> Run:
    run_dir = Path(run_dir)
    if (run_dir / "start.json").exists():
        start = json.loads((run_dir / "start.json").read_text())
    else:  # an exported copy (export()): its meta.yaml
        start = yaml.safe_load((run_dir / "meta.yaml").read_text()) if (run_dir / "meta.yaml").exists() else {}
    explored = (yaml.safe_load((run_dir / "explored_map.yaml").read_text()) or {}).get("objects") or {}
    truth = (yaml.safe_load((run_dir / "true_map.yaml").read_text()) or {}).get("objects") or {}
    run = Run(str(start.get("run", run_dir.name)), str(start.get("detector", "?")), start.get("layout"), explored, truth)
    if (run_dir / "explored_positions.yaml").exists():
        run.explored_3d = yaml.safe_load((run_dir / "explored_positions.yaml").read_text()) or {}
    else:
        run.explored_3d = _sightings_from_frames(run_dir, explored)
    if (run_dir / "true_positions.yaml").exists():
        run.truth_3d = yaml.safe_load((run_dir / "true_positions.yaml").read_text()) or {}
    elif rebuild_truth and start.get("env") == "robocasa":
        run.truth_3d = _rebuilt_truth(start, truth)
        (run_dir / "true_positions.yaml").write_text(yaml.safe_dump(run.truth_3d, sort_keys=False, default_flow_style=None))
    return run


def _sightings_from_frames(run_dir: Path, explored: dict) -> dict:
    """Explored heights from a camera detector's frame logs: the last logged detection of an
    explored object's label at its spot (the tracker logs its running mean, which the map follows)."""
    out = {}
    for path in sorted(glob.glob(str(run_dir / "vision" / "*.json"))):
        for d in json.loads(Path(path).read_text()).get("detections", []):
            for name, e in explored.items():
                if _label(name) == d["label"] and math.dist(e["location"], d["position"][:2]) <= _SAME_SIGHTING:
                    out[name] = [float(v) for v in d["position"]]
    return out


def _rebuilt_truth(start: dict, truth: dict) -> dict:
    """True 3D positions of an older run: its scene built again and reset, then checked against
    the true map it wrote (objects placed elsewhere this time are left out)."""
    import dataclasses

    from chefbot.envs.robocasa.build_map import object_positions
    from chefbot.envs.robocasa.config import RoboCasaConfig
    from chefbot.envs.robocasa.env import RoboCasaEnv

    task = yaml.safe_load(Path(start["task_file"]).read_text()) if start.get("task_file") else {}
    spec = task.get("robocasa") or {}
    config = RoboCasaConfig.load(layout=start.get("layout"), style=start.get("style"), seed=start.get("seed", 0))
    if spec.get("objects"):
        config = dataclasses.replace(config, objects=tuple(dict(o) for o in spec["objects"]))
    env = RoboCasaEnv(config, render=False)
    try:
        rebuilt = object_positions(env)
    finally:
        env.close()
    return {n: p for n, p in rebuilt.items()
            if n in truth and math.dist(p["origin"][:2], truth[n]["location"]) <= _SAME_TRUTH}  # fmt: skip


def export(run: Run, out_dir: str | Path) -> None:
    """A self-contained copy of what this report reads (for regression tests): the objects of
    both maps, both sets of 3D positions, and meta.yaml."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def dump(name: str, data) -> None:
        (out_dir / name).write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=None))

    dump("meta.yaml", {"run": run.name, "detector": run.detector, "layout": run.layout})
    dump("explored_map.yaml", {"objects": run.explored})
    dump("true_map.yaml", {"objects": run.truth})
    dump("explored_positions.yaml", run.explored_3d)
    dump("true_positions.yaml", run.truth_3d)


# -- report ----------------------------------------------------------------------------


def _cm(v: float | None) -> str:
    return "" if v is None else f"{100 * v:.1f}"


def report(runs: list[Run], markdown: bool = False) -> str:
    lines = []
    head = ("| run | detector | layout | matched | xy median | xy p90 | xy max | within 5/10/15 cm | "
            "3D median | z median | xy vs centre median | fixture right |")  # fmt: skip
    lines += [head, "|" + "---|" * 12]
    for run in runs:
        rows = object_errors(run)
        s, s3, sz, sc = (summarize(rows, k) for k in ("xy", "d3", "z", "xy_center"))
        within = "/".join(f"{100 * s.get(f'within_{round(b * 100)}cm', 0):.0f}%" for b in BANDS)
        lines.append(f"| {run.name} | {run.detector} | {run.layout} | {s['matched']}/{s['objects']} | "
                     f"{_cm(s.get('median'))} | {_cm(s.get('p90'))} | {_cm(s.get('max'))} | {within} | "
                     f"{_cm(s3.get('median'))} | {_cm(sz.get('median'))} | {_cm(sc.get('median'))} | "
                     f"{100 * s.get('fixture_ok', 0):.0f}% |")  # fmt: skip
    lines.append("")
    lines.append("Errors in cm. Per object (xy / 3D / reference offset origin to centre, in cm; fixture):")
    for run in runs:
        lines.append(f"\n{run.name} ({run.detector}, layout {run.layout})")
        for r in object_errors(run):
            if r.get("found_as") is None:
                lines.append(f"  {r['object']:<16} missing (in {r['true_in']})")
                continue
            fixture = "ok" if r["in_ok"] else f"in {r['in']}, truly in {r['true_in']}"
            lines.append(f"  {r['object']:<16} xy {_cm(r['xy']):>5}  3D {_cm(r.get('d3')) or '-':>5}  "
                         f"offset {_cm(r.get('offset_xy')) or '-':>5}  {fixture}")  # fmt: skip
    text = "\n".join(lines)
    return text if markdown else text.replace("|", " ").replace(" ---" * 12, "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--rebuild-truth", action="store_true",
                        help="older runs: rebuild the scene for true 3D positions (writes true_positions.yaml)")  # fmt: skip
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--export", type=Path, help="write a self-contained copy of each run under this folder")
    args = parser.parse_args()
    runs = [load_run(r, args.rebuild_truth) for r in args.runs]
    if args.export:
        for run in runs:
            export(run, args.export / f"{run.detector}_layout{run.layout:03d}")
    print(report(runs, args.markdown))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
