"""Run one config over the scene set (spec §17.1).

    python -m experiments.ortho_grasp.run --config experiments/ortho_grasp/configs/base.yaml \
        [--scenes 0-29] [--repeats 3] [--stage prepare,query,evaluate] [--mode batch|sync]

Stages (each re-uses what earlier runs left on disk):
  prepare   scenes (seeded, saved once), the cup's geometry per scene, and the
            rendered inputs (shared by every config with the same render).
  query     model calls: sync, or batch submit (then `collect`). Sequential
            views query in two rounds: the side-view call needs the top answer.
  collect   poll submitted batches and cache finished results.
  evaluate  parse -> resolve -> E1 -> E2, with overlays, per scene and repeat.
  e3        execute in the simulator (`--stage e3`).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import yaml

from experiments.ortho_grasp import scenes as scene_set
from experiments.ortho_grasp.frames import MAX_OPEN, pose_to_world, tcp_pose
from experiments.ortho_grasp.observe import CupGeometry, observe
from experiments.ortho_grasp.query import models as M
from experiments.ortho_grasp.query import schemas
from experiments.ortho_grasp.render import plots
from experiments.ortho_grasp.resolve import (
    Resolved, heuristic, lift, quat_to_matrix, resolve_p1, resolve_p2, resolve_pose, resolve_waypoints,
)  # fmt: skip

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


# -- configuration -------------------------------------------------------------------


def load_config(path: Path) -> dict:
    exp = yaml.safe_load((HERE / "configs" / "experiment.yaml").read_text())
    models = yaml.safe_load((HERE / "configs" / "models.yaml").read_text())
    base = yaml.safe_load((HERE / "configs" / "base.yaml").read_text())
    cfg = {**base, **yaml.safe_load(Path(path).read_text())}  # every config is base plus the axis it varies
    cfg["experiment"] = exp
    cfg["model_spec"] = models.get(cfg["model"]) if cfg["representation"] not in ("r_gt", "r_heur") else None
    cfg["results"] = REPO / exp["results_dir"]
    return cfg


def parse_range(text: str | None, n: int) -> list[int]:
    if not text:
        return list(range(n))
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


class Lazy:
    """The simulator, built on first use (prepare, R-GT, E2's IK, E3)."""

    def __init__(self, exp: dict):
        self.exp, self._world = exp, None

    def __call__(self):
        if self._world is None:
            from experiments.ortho_grasp.sim import World

            self._world = World({**self.exp, "results_dir": str(REPO / self.exp["results_dir"])})
        return self._world

    def close(self):
        if self._world is not None:
            self._world.close()


# -- scenes, geometry, inputs ---------------------------------------------------------------


def ensure_scenes(cfg: dict, world: Lazy) -> list[scene_set.Scene]:
    path = cfg["results"] / "scenes.json"
    if path.exists():
        return scene_set.load(path)
    w = world()
    exp = cfg["experiment"]

    def reachable(xy) -> bool:
        pose = tcp_pose(np.array([xy[0], xy[1], exp["counter_z"] + 0.08]), [0, 0, -1], [0, 1, 0])
        return ik_robot(w, pose, None) is not None

    scenes = scene_set.generate(exp, reachable)
    scene_set.save(scenes, path)
    return scenes


def ik_robot(w, pose_robot: np.ndarray, seed) -> np.ndarray | None:
    arm = w.arm.frames.pose_to_arm_frame(w.base, pose_to_world(pose_robot, w.base))
    return w.arm.ik.solve_pose(arm, seed=w.arm.ready if seed is None else np.asarray(seed))


def geometry_path(cfg: dict, scene) -> Path:
    return cfg["results"] / "geometry" / f"{scene.name}.npz"


def load_geometry(cfg: dict, scene) -> CupGeometry:
    d = np.load(geometry_path(cfg, scene))
    hulls = np.split(d["hull_eq"], d["hull_split"])
    return CupGeometry(d["full"], d["partial"], hulls, float(d["counter_z"]), float(d["rim_z"]), float(d["bottom_z"]))


def plot_margin(cfg: dict) -> float:
    return float(cfg.get("plot_margin") or cfg["experiment"]["plot_margin"])


def input_dir(cfg: dict, scene) -> Path:
    key = "b0" if cfg["representation"] == "b0" else f"{cfg['render']}_{cfg['geometry']}"
    if cfg["representation"] != "b0" and plot_margin(cfg) != float(cfg["experiment"]["plot_margin"]):
        key += f"_m{plot_margin(cfg):.2f}"
    return cfg["results"] / "inputs" / key / scene.name


def prepare(cfg: dict, scenes, world: Lazy) -> None:
    exp = cfg["experiment"]
    for scene in scenes:
        gpath, idir = geometry_path(cfg, scene), input_dir(cfg, scene)
        need_inputs = cfg["representation"] not in ("r_gt", "r_heur") and not (idir / "done").exists()
        if gpath.exists() and not need_inputs:
            continue
        w = world()
        w.apply(scene)
        if not gpath.exists():
            g = observe(w, scene, np.random.default_rng(scene.seed))
            gpath.parent.mkdir(parents=True, exist_ok=True)
            np.savez(gpath, full=g.full, partial=g.partial, hull_eq=np.concatenate(g.hulls),
                     hull_split=np.cumsum([len(h) for h in g.hulls])[:-1], counter_z=g.counter_z, rim_z=g.rim_z,
                     bottom_z=g.bottom_z, cup_pose=w.cup_pose_robot(scene))  # fmt: skip
        if need_inputs:
            render_inputs(cfg, scene, w, load_geometry(cfg, scene), idir)
        print(f"prepared {scene.name}", flush=True)


def render_inputs(cfg: dict, scene, w, g: CupGeometry, idir: Path) -> None:
    idir.mkdir(parents=True, exist_ok=True)
    if cfg["representation"] == "b0":
        from experiments.ortho_grasp.render import camera

        info = camera.capture(w, idir)
        (idir / "b0.json").write_text(json.dumps(info, indent=1))
    else:
        specs = plots.specs_for(g.box, plot_margin(cfg), cfg["image_size"])
        for view, spec in specs.items():
            if cfg["render"] == "v4":
                from experiments.ortho_grasp.render import ortho_sim

                mid = (g.box[0] + g.box[1]) / 2
                img, mask = ortho_sim.render(w, spec, w.cup_root(scene), mid[2] if view == "top" else mid[1])
                ortho_sim.check_alignment(spec, mask, g.full, np.random.default_rng(scene.seed))
                plots.draw(spec, idir / f"{view}.png", image=img, counter_z=g.counter_z, marks="ABF")
            else:
                pts = g.full if cfg["geometry"] == "full" else g.partial
                plots.draw(spec, idir / f"{view}.png", points=pts, counter_z=g.counter_z, marks="ABF",
                           color_range=(g.bottom_z, g.rim_z))  # fmt: skip
    (idir / "done").write_text("")


def legend_marks(cfg: dict) -> str:
    return "ABF" if cfg["representation"] == "p1" else "AB"


# -- requests ---------------------------------------------------------------------------------


def specs_of(cfg: dict, scene) -> dict[str, plots.PlotSpec]:
    idir = input_dir(cfg, scene)
    return {v: plots.PlotSpec.load(idir / f"{v}.json") for v in ("top", "side")}


def plot_images(cfg: dict, scene) -> dict[str, Path]:
    """The plots for this config: the shared render, redrawn if the legend differs (P2 has no F)."""
    idir = input_dir(cfg, scene)
    if legend_marks(cfg) == "ABF":
        return {v: idir / f"{v}.png" for v in ("top", "side")}
    out = {}
    g = load_geometry(cfg, scene)
    specs = specs_of(cfg, scene)
    for v, spec in specs.items():
        p = idir / f"{v}_ab.png"
        if not p.exists():
            if cfg["render"] == "v4":
                raise RuntimeError("P2 with V4 is not in the matrix")  # would need the render kept; not needed in v1
            pts = g.full if cfg["geometry"] == "full" else g.partial
            plots.draw(spec, p, points=pts, counter_z=g.counter_z, marks="AB", color_range=(g.bottom_z, g.rim_z))
        out[v] = p
    return out


def model_request(cfg: dict, text: str, images: list[Path], repeat: int, meta: dict) -> M.Request:
    ms = cfg["model_spec"]
    temperature = ms["temperature"] if "temperature" in ms else cfg["temperature"]
    return M.Request(ms["provider"], ms["model"], text, [str(p) for p in images], temperature, int(cfg["max_tokens"]), repeat, meta=meta)


def first_requests(cfg: dict, scene, repeat: int) -> M.Request:
    meta = {"config": cfg["name"], "scene": scene.name, "repeat": repeat, "call": 1}
    if cfg["representation"] == "b0":
        info = json.loads((input_dir(cfg, scene) / "b0.json").read_text())
        text = schemas.prompt_b0(cfg, info["camera_text"], cfg["experiment"]["counter_z"], info["tcp_position"], info["tcp_quaternion"])
        return model_request(cfg, text, info["images"], repeat, meta)
    imgs = plot_images(cfg, scene)
    text = schemas.prompt(cfg, cfg["image_size"])
    images = [imgs["top"]] if cfg["views"] == "seq" else [imgs["top"], imgs["side"]]
    return model_request(cfg, text, images, repeat, meta)


def second_request(cfg: dict, scene, repeat: int, first: dict, run_dir: Path) -> M.Request | None:
    """C-seq: the side view with guide lines at the top view's x values."""
    specs = specs_of(cfg, scene)
    try:
        top = schemas.parse_marks(cfg, first["text"], specs)["top"]
    except schemas.ParseError:
        return None
    guides = {k: float(v[0]) for k, v in top.items()}
    g = load_geometry(cfg, scene)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "side_guides.png"
    if cfg["render"] == "v4":
        raise RuntimeError("sequential views with V4 are not in the matrix")
    pts = g.full if cfg["geometry"] == "full" else g.partial
    plots.draw(specs["side"], path, points=pts, counter_z=g.counter_z, marks=legend_marks(cfg),
               color_range=(g.bottom_z, g.rim_z), guides=guides)  # fmt: skip
    previous = {"top": {k: [round(float(x), 4) for x in v] for k, v in top.items()}}
    text = schemas.prompt_seq_side(cfg, cfg["image_size"], previous, guides)
    return model_request(cfg, text, [path], repeat, {"config": cfg["name"], "scene": scene.name, "repeat": repeat, "call": 2})


def run_dir_of(cfg: dict, scene, repeat: int) -> Path:
    return cfg["results"] / cfg["name"] / f"{scene.name}_{repeat}"


def query(cfg: dict, scenes, repeats: int, mode: str, cache: M.Cache) -> None:
    if cfg["representation"] in ("r_gt", "r_heur"):
        return
    batch_dir = cfg["results"] / "batches"
    firsts = [first_requests(cfg, s, r) for s in scenes for r in range(repeats)]
    _dispatch(firsts, mode, cache, batch_dir, cfg)
    if cfg["views"] == "seq":
        seconds = []
        for s in scenes:
            for r in range(repeats):
                first = cache.get(first_requests(cfg, s, r).key)
                if first is not None:
                    req = second_request(cfg, s, r, first, run_dir_of(cfg, s, r))
                    if req is not None:
                        seconds.append(req)
        _dispatch(seconds, mode, cache, batch_dir, cfg)


def _dispatch(reqs, mode, cache, batch_dir, cfg) -> None:
    if not reqs:
        return
    if mode == "sync":
        done = M.run_sync(reqs, cache)
        errors = [d for d in done if d.get("error")]
        print(f"sync: {len(done)} responses, {len(errors)} errors" + (f" (first: {errors[0]['error']})" if errors else ""), flush=True)
    else:
        ids = M.submit_batches(reqs, cache, batch_dir, cfg["experiment"].get("gcs_bucket"))
        print(f"batch: submitted {len(ids)} batch(es) for {len([r for r in reqs if cache.get(r.key) is None])} uncached requests", flush=True)


# -- evaluation ---------------------------------------------------------------------------------


def resolve_one(cfg: dict, scene, repeat: int, cache: M.Cache, world: Lazy) -> tuple[Resolved, dict, dict]:
    """(resolved, response record(s), marks per view for the overlays)."""
    rep, g = cfg["representation"], load_geometry(cfg, scene)
    if rep == "r_heur":
        return heuristic(g.box, g.rim_z), {}, {}
    if rep == "r_gt":
        return gt_reference(cfg, scene, world, g), {}, {}
    first = cache.get(first_requests(cfg, scene, repeat).key)
    if first is None:
        return Resolved(False, "no_response"), {}, {}
    responses = {"call1": first}
    if first.get("error") or not first.get("text"):
        return Resolved(False, "api_error"), responses, {}
    try:
        if rep == "b0":
            p, q, wdt = schemas.parse_b0(first["text"])
            return resolve_pose(p, quat_to_matrix(q), wdt), responses, {}
        specs = specs_of(cfg, scene)
        parsed = schemas.parse_marks(cfg, first["text"], specs)
        n = schemas.names(cfg)
        if cfg["views"] == "seq":
            second_req = second_request(cfg, scene, repeat, first, run_dir_of(cfg, scene, repeat))
            second = cache.get(second_req.key) if second_req is not None else None
            if second is None:
                return Resolved(False, "no_response_call2"), responses, {"top": parsed["top"]}
            responses["call2"] = second
            zs = schemas.parse_seq_side(cfg, second["text"], specs["side"])
            parsed["side"] = {k: (float(parsed["top"][k][0]), z) for k, z in zs.items()}
            points, dis = lift(parsed["top"], zs, n)
            marks = {"top": parsed["top"], "side": parsed["side"]}
        elif cfg["trajectory"] == "t3":
            per, grippers, marks = [], [], {"top": {}, "side": {}}
            for i, wp in enumerate(parsed["waypoints"]):
                pts, dis = lift(wp["top"], wp["side"], n)
                per.append(resolve_p1(pts, dis) if rep == "p1" else resolve_p2(pts, dis, _cloud(cfg, g), g.counter_z))
                grippers.append(wp["gripper"])
                for v in ("top", "side"):
                    marks[v].update({f"{k}{i + 1}": c for k, c in wp[v].items()})
            return resolve_waypoints(per, grippers), responses, marks
        else:
            points, dis = lift(parsed["top"], parsed["side"], n)
            marks = parsed
    except schemas.ParseError as e:
        return Resolved(False, f"parse:{e}"), responses, {}
    r = resolve_p1(points, dis) if rep == "p1" else resolve_p2(points, dis, _cloud(cfg, g), g.counter_z)
    return r, responses, marks


def _cloud(cfg: dict, g: CupGeometry) -> np.ndarray:
    return g.full if cfg["geometry"] == "full" else g.partial


def gt_reference(cfg: dict, scene, world: Lazy, g: CupGeometry) -> Resolved:
    """R-GT: a ground-truth rim grasp (fingers astride the wall nearest the robot,
    from the cup's mesh), the first around the rim the arm reaches with the
    gripper clear. The repo proposer's own rim grasp assumes a container wider
    than the gripper's stroke and returns none for these cups."""
    from experiments.ortho_grasp.resolve import gripper_clear, rim_grasps

    w = world()
    for r in rim_grasps(g.full, g.rim_z):
        if ik_robot(w, r.waypoints[0]["pose"], None) is None or ik_robot(w, r.pose, None) is None:
            continue
        if gripper_clear(r.pose, min(MAX_OPEN, r.width + 0.01), g.full, g.counter_z, step=0.005, touch_last=0.01):
            return r
    return Resolved(False, "no_gt_grasp")


def evaluate(cfg: dict, scenes, repeats: int, cache: M.Cache, world: Lazy) -> list[dict]:
    from experiments.ortho_grasp.evaluate import e1_geometric, e2_kinematic

    rows = []
    reps = 1 if cfg["representation"] in ("r_gt", "r_heur") else repeats
    for scene in scenes:
        g = load_geometry(cfg, scene)
        for repeat in range(reps):
            out = run_dir_of(cfg, scene, repeat)
            out.mkdir(parents=True, exist_ok=True)
            r, responses, marks = resolve_one(cfg, scene, repeat, cache, world)
            metrics = {"config": cfg["name"], "scene": scene.name, "repeat": repeat, "cup_model": scene.cup_model,
                       "distractors": len(scene.distractors), "e0_parse": r.reason is None or not str(r.reason).startswith(("parse", "no_response", "api_error")),
                       "resolved": r.ok, "reason": r.reason, **{f"flag_{k}": v for k, v in r.flags.items()}, **r.metrics}  # fmt: skip
            for name, resp in responses.items():
                for k in ("input_tokens", "output_tokens", "cost", "latency_s"):
                    metrics[f"{name}_{k}"] = resp.get(k)
            fixed = flange_corrected(cfg, r, g, world)
            if fixed is not None:
                metrics.update(score_fixed(fixed, g, world))
                (out / "resolved_fixed.json").write_text(json.dumps(fixed.to_json(), indent=1, default=str))
            if r.ok:
                A, B = r.fingertips
                e1 = e1_geometric.evaluate(A, B, r.width, g)
                e2 = e2_kinematic.evaluate(r, g, lambda pose, seed: ik_robot(world(), pose, seed), e1["e1_valid"])
                metrics.update(e1)
                metrics.update({k: v for k, v in e2.items() if not k.startswith("q_")})
                r.metrics.update({"q_pre": e2["q_pre"], "q_grasp": e2["q_grasp"]})
            else:
                metrics.update({"e1_valid": False, "e2_valid": False})
            (out / "response.json").write_text(json.dumps(responses, indent=1, default=str))
            (out / "resolved.json").write_text(json.dumps(r.to_json(), indent=1, default=str))
            (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
            write_inputs(cfg, scene, repeat, out, cache)
            draw_overlays(cfg, scene, out, r, marks, metrics)
            rows.append(metrics)
    print(f"evaluated {len(rows)}: e2_valid {sum(bool(m.get('e2_valid')) for m in rows)}/{len(rows)}", flush=True)
    return rows


def flange_corrected(cfg: dict, r: Resolved, g: CupGeometry, world: Lazy) -> Resolved | None:
    """P1 and B0 (T1): the model's fingers kept, the flange moved until the grasp is
    valid against what the model saw (see resolve.correct_flange). None: not applicable."""
    from experiments.ortho_grasp.resolve import correct_flange

    if cfg["trajectory"] != "t1" or cfg["representation"] not in ("p1", "b0"):
        return None
    if cfg["representation"] == "p1":
        if not all(k in r.points for k in ("A", "B", "F")):
            return None
        A, B, F = (np.array(r.points[k]) for k in ("A", "B", "F"))
    else:
        if r.pose is None:
            return None
        A, B = r.fingertips
        F = r.pose[:3, 3] - 0.156 * r.pose[:3, 2]
    cloud = g.full if cfg.get("geometry", "full") == "full" or cfg["representation"] == "b0" else g.partial

    def reachable(rr: Resolved) -> bool:
        q = ik_robot(world(), rr.waypoints[0]["pose"], None)
        return q is not None and ik_robot(world(), rr.pose, q) is not None

    return correct_flange(A, B, F, cloud, g.counter_z, reachable)


def score_fixed(fixed: Resolved, g: CupGeometry, world: Lazy) -> dict:
    from experiments.ortho_grasp.evaluate import e1_geometric, e2_kinematic

    out = {"fix_resolved": fixed.ok, "fix_reason": fixed.reason}
    if not fixed.ok:
        return {**out, "fix_e1_valid": False, "fix_e2_valid": False}
    A, B = fixed.fingertips
    e1 = e1_geometric.evaluate(A, B, fixed.width, g)
    e2 = e2_kinematic.evaluate(fixed, g, lambda pose, seed: ik_robot(world(), pose, seed), e1["e1_valid"])
    out.update({f"fix_{k}": v for k, v in e1.items()})
    out.update({f"fix_{k}": v for k, v in e2.items() if not k.startswith("q_")})
    out.update({f"fix_{k}": v for k, v in fixed.metrics.items()})
    return out


def write_inputs(cfg: dict, scene, repeat: int, out: Path, cache: M.Cache) -> None:
    """inputs/: the exact images sent, plot specs, rendered prompt(s), the config."""
    d = out / "inputs"
    d.mkdir(exist_ok=True)
    snapshot = {k: v for k, v in cfg.items() if k not in ("experiment", "results")}
    (d / "config.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(snapshot, default=str))))
    if cfg["representation"] in ("r_gt", "r_heur"):
        return
    req = first_requests(cfg, scene, repeat)
    (d / "prompt_call1.txt").write_text(req.text)
    files = list(req.images)
    if cfg["representation"] != "b0":
        files += [str(input_dir(cfg, scene) / f"{v}.json") for v in ("top", "side")]
    if (out / "side_guides.png").exists():
        files.append(str(out / "side_guides.png"))
    for f in files:
        dst = d / Path(f).name
        if not dst.exists():
            try:
                os.link(f, dst)
            except OSError:
                shutil.copy(f, dst)


def draw_overlays(cfg: dict, scene, out: Path, r: Resolved, marks: dict, metrics: dict) -> None:
    """The model's marks (A, B, F; M; the corrected flange F*) on the plots it saw.
    B0 and the references have no marks of their own: their A, B, F are drawn on
    the full-cloud plots."""
    idir = input_dir(cfg, scene) if cfg["representation"] != "b0" else None
    if idir is None or not (idir / "top.json").exists():
        idir = cfg["results"] / "inputs" / "v1_full" / scene.name
        if not (idir / "top.json").exists():
            return
    points3d = {}
    if not marks and r.pose is not None:  # B0, references: from the pose
        A, B = r.fingertips
        points3d = {"A": A, "B": B, "F": r.pose[:3, 3] - 0.156 * r.pose[:3, 2]}
    if r.ok and cfg["trajectory"] == "t1" and r.pose is not None:
        points3d["M"] = r.pose[:3, 3]
    fixed = out / "resolved_fixed.json"
    if fixed.exists():
        fx = json.loads(fixed.read_text())
        if fx.get("ok") and "F" in fx.get("points", {}) and (fx.get("metrics") or {}).get("flange_correction_deg", 0.0) > 1.0:
            points3d["F*"] = np.array(fx["points"]["F"])  # only when the correction moved it
    note = f"{cfg['name']} {scene.name}: " + ("e2_valid" if metrics.get("e2_valid") else f"fail ({_why(metrics)})")
    if "fix_e2_valid" in metrics:
        note += f" | flange corrected: {'e2_valid' if metrics.get('fix_e2_valid') else 'fail'}"
    for v in ("top", "side"):
        spec = plots.PlotSpec.load(idir / f"{v}.json")
        bg = out / "side_guides.png" if v == "side" and (out / "side_guides.png").exists() else idir / f"{v}.png"
        view_marks = {k: np.asarray(c) for k, c in (marks.get(v) or {}).items()}
        view_marks.update({k: spec.project(np.asarray(p)[None])[0] for k, p in points3d.items()})
        plots.overlay(spec, bg, out / f"overlay_{v}.png", view_marks, bool(metrics.get("e2_valid")), note)


CONTACT_TOLERANCE = 0.005  # m per finger: marks on (or just in) the cup's surface count as touching, not inside


def widened(r: Resolved, tol: float = CONTACT_TOLERANCE) -> Resolved | None:
    """The same grasp with each finger moved `tol` outward along the closing line
    (the gripper opened a little wider before closing), capped at 85 mm."""
    if not r.ok or r.pose is None:
        return None
    out = Resolved(True, None, dict(r.points), r.pose.copy(), min(MAX_OPEN, r.width + 2 * tol), [], dict(r.flags), dict(r.metrics))
    out.waypoints = [dict(w) for w in r.waypoints]
    return out


def rescore(cfg: dict, scenes, repeats: int, world: Lazy) -> None:
    """tol_*: E1/E2 with contact tolerance; tolfix_*: the same on the flange-corrected grasp."""
    from experiments.ortho_grasp.evaluate import e1_geometric, e2_kinematic
    from experiments.ortho_grasp.resolve import Resolved as R

    def load(path: Path) -> Resolved | None:
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        r = R(d["ok"], d["reason"], d["points"], None if d["pose"] is None else np.array(d["pose"]), d["width"])
        r.waypoints = [{**w, "pose": np.array(w["pose"])} for w in d["waypoints"]]
        return r

    reps = 1 if cfg["representation"] in ("r_gt", "r_heur") else repeats
    n_tol = n_fix = 0
    for scene in scenes:
        g = load_geometry(cfg, scene)
        for repeat in range(reps):
            out = run_dir_of(cfg, scene, repeat)
            if not (out / "metrics.json").exists():
                continue
            metrics = json.loads((out / "metrics.json").read_text())
            for prefix, path in (("tol_", out / "resolved.json"), ("tolfix_", out / "resolved_fixed.json")):
                w = widened(load(path)) if load(path) is not None else None
                if w is None:
                    metrics.update({f"{prefix}e1_valid": False, f"{prefix}e2_valid": False})
                    continue
                A, B = w.fingertips
                e1 = e1_geometric.evaluate(A, B, w.width, g)
                e2 = e2_kinematic.evaluate(w, g, lambda pose, seed: ik_robot(world(), pose, seed), e1["e1_valid"])
                metrics.update({f"{prefix}{k}": v for k, v in e1.items()})
                metrics.update({f"{prefix}{k}": v for k, v in e2.items() if not k.startswith("q_")})
            n_tol += bool(metrics.get("tol_e2_valid"))
            n_fix += bool(metrics.get("tol_e2_valid") or metrics.get("tolfix_e2_valid"))
            (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    print(f"rescored: tol e2_valid {n_tol}, with flange correction {n_fix}", flush=True)


def redraw_overlays(cfg: dict, scenes, repeats: int, cache: M.Cache) -> None:
    """Overlays again from what evaluate saved (no IK, no model calls)."""
    from experiments.ortho_grasp.resolve import Resolved as R

    reps = 1 if cfg["representation"] in ("r_gt", "r_heur") else repeats
    for scene in scenes:
        for repeat in range(reps):
            out = run_dir_of(cfg, scene, repeat)
            if not (out / "resolved.json").exists():
                continue
            d = json.loads((out / "resolved.json").read_text())
            r = R(d["ok"], d["reason"], d["points"], None if d["pose"] is None else np.array(d["pose"]), d["width"])
            metrics = json.loads((out / "metrics.json").read_text())
            marks = {}
            if cfg["representation"] in ("p1", "p2"):
                _, _, marks = resolve_one(cfg, scene, repeat, cache, None)  # the marks as the model gave them
            draw_overlays(cfg, scene, out, r, marks, metrics)


def _why(m: dict) -> str:
    if not m.get("resolved"):
        return str(m.get("reason"))
    for k in ("straddle", "clear_tips", "width_ok", "height_ok", "ik_ok", "collision_ok"):
        if m.get(k) is False:
            return k
    return "?"


# -- main ---------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one ortho-grasp config over the scene set.")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--scenes")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--stage", default="prepare,query,collect,evaluate")
    ap.add_argument("--mode", choices=["batch", "sync"], default="batch")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached responses (re-query)")
    args = ap.parse_args()
    M.load_env(REPO / ".env")
    cfg = load_config(args.config)
    world = Lazy(cfg["experiment"])
    cache = M.Cache(cfg["results"] / ("cache_nocache" if args.no_cache else "cache"))
    try:
        all_scenes = ensure_scenes(cfg, world)
        scenes = [all_scenes[i] for i in parse_range(args.scenes, len(all_scenes))]
        repeats = args.repeats or int(cfg["repeats"])
        stages = args.stage.split(",")
        if "prepare" in stages:
            prepare(cfg, scenes, world)
        if "query" in stages:
            query(cfg, scenes, repeats, args.mode, cache)
        if "collect" in stages and args.mode == "batch":
            print("batches:", M.collect_batches(cache, cfg["results"] / "batches"), flush=True)
            if cfg["views"] == "seq" and "query" in stages:
                query(cfg, scenes, repeats, args.mode, cache)  # the second round, for first calls now answered
        if "evaluate" in stages:
            evaluate(cfg, scenes, repeats, cache, world)
        if "rescore" in stages:
            rescore(cfg, scenes, repeats, world)
        if "overlays" in stages:
            redraw_overlays(cfg, scenes, repeats, cache)
        if "e3" in stages:
            from experiments.ortho_grasp.evaluate import e3_sim

            e3_sim.run(cfg, scenes, world, run_dir_of, load_geometry)
    finally:
        world.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
