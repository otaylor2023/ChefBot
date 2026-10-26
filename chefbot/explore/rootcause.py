"""Why a camera detector misses or misplaces objects: a sighting table, per frame and per object.

    python -m chefbot.explore.rootcause runs/<run> [runs/<run> ...] --out <folder> [--owl-scores]

Offline analysis of finished runs; it reads the truth, the agent never did. A run's logs hold the
detector's frames (vision/NNN.png, .json: what it named, where that put things) but not the camera
pose, depth or what the robot's own geometry hid. The scripted route is deterministic (door
handles come from the scene's fixture geometry under every detector), so this replays the run's
tool calls in the same scene with the exact detector, and at each call that produced a detector
frame renders the same camera: RGB-D and a segmentation image. That gives, per true object in each
frame: visible pixels, its box, distance, viewing angle, whether it touches the image edge; and,
from the frame's log, the detector's outcome for it:

  kept         a detection of its label within 0.3 m of it (with the position error)
  dropped      named right in the image, but no detection came of it (edge filter, depth on the
               background, the height / footprint filters, or a merge)
  wrong label  a box over it named something else
  not named    in view, no box over it

A logged frame is matched to the replayed view it looks like (mean absolute difference under
_SAME_IMAGE); frames that match no replayed view (a scene that differs, e.g. an object placed
elsewhere by a later spawn change) are left out and counted.

--owl-scores also runs OWLv2 on every saved frame and records, per object in view, its best score
for the right label over the boxes on it, below the threshold too.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import yaml

_EDGE = 8  # px, as the vision detector's edge filter
_KEPT = 0.3  # m: a detection of the label this close to the object is a sighting of it
_SAME_IMAGE = 3.0  # mean abs difference (0..255) under which a replayed frame is the logged one


def _box_overlap(box, bbox) -> float:
    """Share of `bbox` (the object's visible pixels' box) covered by `box`."""
    x0, y0, x1, y1 = max(box[0], bbox[0]), max(box[1], bbox[1]), min(box[2], bbox[2]), min(box[3], bbox[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return inter / area


def replay(run_dir: Path) -> list[dict]:
    """One row per (detector frame, true object in the camera's view)."""
    from PIL import Image

    from chefbot.agent.config import AgentConfig
    from chefbot.agent.tools.registry import ToolRegistry
    from chefbot.agent.world.map import WorldMap
    from chefbot.envs.robocasa.build_map import object_positions
    from chefbot.envs.robocasa.detector import FIXTURE_LABELS, PART_RULES, scene_objects
    from chefbot.explore.vision import _CONFIG, _VOCAB, normalize
    from chefbot.interface.types import Pose2D
    from chefbot.scripts.explore import EXPLORE, REPO, robocasa_world

    start = json.loads((run_dir / "start.json").read_text())
    frames = sorted(glob.glob(str(run_dir / "vision" / "*.json")))
    logs = [json.loads(Path(f).read_text()) for f in frames]
    camera = next((log["camera"] for log in logs if log.get("camera")), _CONFIG["vision"]["camera"])
    truth = (yaml.safe_load((run_dir / "true_map.yaml").read_text()) or {}).get("objects") or {}
    task_file = Path(start["task_file"])
    task_file = task_file if task_file.is_absolute() else REPO / task_file
    world = robocasa_world(task_file, int(start.get("seed", 0)), Path(tempfile.mkdtemp()),
                           "groundtruth", start.get("layout"), start.get("style"))  # fmt: skip
    env = world.stack.env
    explore_cfg = yaml.safe_load((EXPLORE / "config.yaml").read_text())
    vocab = yaml.safe_load((EXPLORE / explore_cfg["vocab_file"]).read_text())
    import dataclasses

    agent_cfg = dataclasses.replace(AgentConfig.load(), default_vocab=tuple(dict.fromkeys([*vocab["parts"], *vocab["objects"]])),
                                    search_default_vocab=False)  # fmt: skip
    registry = ToolRegistry(world.robot, world.detector, WorldMap.from_dict(world.start), agent_cfg)
    fixture_labels = set(FIXTURE_LABELS.values()) | {r[3] for r in PART_RULES} | set(_VOCAB["parts"])
    synonyms, words = dict(_CONFIG["vision"].get("synonyms") or {}), set(_VOCAB["objects"])

    # The scene's objects, named as the true map names them, with their geoms (all groups: what is drawn).
    positions = object_positions(env)
    model = env.model
    objects = []
    for _, label, body in scene_objects(env):
        origin = env.data.xpos[body]
        name = min(positions, key=lambda n: float(np.linalg.norm(np.array(positions[n]["origin"]) - origin)))
        root = model.body_rootid[body]
        geoms = np.array([g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == root])
        objects.append((name, label, body, geoms))

    from chefbot.envs.robocasa.objects import object_points

    rows, frame, skipped = [], 0, 0
    actions = [json.loads(line) for line in (run_dir / "run.jsonl").read_text().splitlines()]
    for record in (r for r in actions if r.get("kind") == "action"):
        tool, args = record["tool"], record["args"]
        if tool == "detect" and frame < len(logs):
            # Any detect may have called the model: even one for handles also asks about the
            # objects on the agent's map nearby. Frames are matched by what they show.
            if True:
                # The detector logged a frame for each model call; a call is a view that changed (in the
                # run, physics jitter can change it at the same spot too). So: every logged frame that
                # looks like this view is this view.
                rgbd = env.get_camera_rgbd(camera)

                def diff_to(i: int) -> float:
                    saved = np.asarray(Image.open(frames[i].replace(".json", ".png")).convert("RGB"), dtype=float)
                    return float(np.abs(saved - rgbd.rgb).mean())

                ahead = [i for i in range(frame, min(frame + 4, len(logs))) if diff_to(i) <= _SAME_IMAGE]
                if ahead:
                    skipped += ahead[0] - frame  # logged frames no replayed view looks like
                    frame = ahead[0]
                while frame < len(logs):
                    diff = diff_to(frame)
                    if diff > _SAME_IMAGE:
                        break
                    rows += _sightings(env, camera, rgbd, objects, logs[frame], frame + 1, diff, truth,
                                       object_points, fixture_labels, synonyms, words, record)  # fmt: skip
                    frame += 1
        if tool in ("go_to", "align", "open", "close", "detect", "nearby", "wait", "stop"):
            registry.dispatch(tool, args)
            pose = (record.get("data") or {}).get("pose")
            if pose is not None:  # the run's motion ran in real time: stand exactly where it stopped
                env.teleport_base(Pose2D(float(pose["x"]), float(pose["y"]), math.radians(float(pose["yaw_deg"]))))
    world.close()
    for r in rows:
        r.update(run=run_dir.name, detector=start.get("detector"), layout=start.get("layout"), frames_logged=len(logs),
                 frames_replayed=frame - skipped)  # fmt: skip
    return rows


def _segmentation(env, camera: str, h: int, w: int) -> np.ndarray:
    def render():
        with env.lock:
            return env.suite.sim.render(camera_name=camera, width=w, height=h, segmentation=True)

    import mujoco

    seg = env._render_thread.submit(render).result()[::-1]
    return np.where(seg[..., 0] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 1], -1)


def _sightings(env, camera, rgbd, objects, log, frame, diff, truth, object_points, fixture_labels, synonyms, words,
               record) -> list[dict]:  # fmt: skip
    from chefbot.explore.vision import normalize

    h, w = rgbd.depth.shape
    seg = _segmentation(env, camera, h, w)
    cam = rgbd.cam_to_world
    world_to_cam = np.linalg.inv(cam)
    origin, axis = cam[:3, 3], cam[:3, 2]
    named = [{"label": normalize(i["label"], synonyms, words), "box": i["box"], "score": i.get("score")} for i in log["vlm"]]
    dets = log.get("detections", [])
    out = []
    for name, label, body, geoms in objects:
        points = object_points(env, body)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        size = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))  # bounding box diagonal, m
        p = world_to_cam @ np.append(center, 1.0)
        in_front = p[2] > 0.05
        u, v = (rgbd.intrinsics @ (p[:3] / p[2]))[:2] if in_front else (np.nan, np.nan)
        in_frustum = bool(in_front and 0 <= u < w and 0 <= v < h)
        mask = np.isin(seg, geoms)
        visible = int(mask.sum())
        if not in_frustum and visible == 0:
            continue
        # Its projection if nothing were in the way: the box of its points in front of the camera.
        cp = (world_to_cam @ np.c_[points, np.ones(len(points))].T).T
        cp = cp[cp[:, 2] > 0.05]
        proj = (rgbd.intrinsics @ (cp[:, :3] / cp[:, 2:3]).T).T[:, :2] if len(cp) else np.zeros((0, 2))
        proj_area = float(np.ptp(np.clip(proj[:, 0], 0, w)) * np.ptp(np.clip(proj[:, 1], 0, h))) if len(proj) else 0.0
        direction = center - origin
        distance = float(np.linalg.norm(direction))
        row = {"frame": frame, "image_diff": round(diff, 2), "camera": camera, "object": name, "label": label,
               "true_in": (truth.get(name) or {}).get("in"), "size_m": round(size, 3), "distance_m": round(distance, 3),
               "angle_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, float(axis @ direction) / distance)))), 1),
               "in_frustum": in_frustum, "visible_px": visible, "projected_px": round(proj_area),
               "visible_share": round(visible / proj_area, 3) if proj_area > 0 else 0.0, "base_pose": record.get("data", {}).get("pose")}  # fmt: skip
        if visible:
            ys, xs = np.nonzero(mask)
            bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
            row["_bbox"] = bbox
            row.update(box_h_px=round(bbox[3] - bbox[1]), box_w_px=round(bbox[2] - bbox[0]),
                       at_edge=bool(bbox[0] < _EDGE or bbox[1] < _EDGE or bbox[2] > w - _EDGE or bbox[3] > h - _EDGE))  # fmt: skip
            on_it = [n for n in named if _box_overlap(n["box"], bbox) >= 0.3]
            right = [n for n in on_it if n["label"] == label]
            wrong = [n["label"] for n in on_it if n["label"] != label and n["label"] not in fixture_labels]
            kept = [d for d in dets if d["label"] == label and np.linalg.norm(np.array(d["position"]) - center) <= _KEPT]
            if kept:
                d = min(kept, key=lambda d: np.linalg.norm(np.array(d["position"]) - center))
                err = np.array(d["position"]) - center
                row.update(outcome="kept", error_m=round(float(np.linalg.norm(err)), 3), error_xy_m=round(float(np.linalg.norm(err[:2])), 3))
            elif right:
                box = right[0]["box"]
                cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                edge = min(box[0], box[1]) < _EDGE or box[2] > w - 1 - _EDGE or box[3] > h - 1 - _EDGE
                hit = bool(np.isin(seg[min(h - 1, max(0, cy)), min(w - 1, max(0, cx))], geoms))
                row.update(outcome="dropped", drop_reason="edge filter" if edge else ("depth hit the object" if hit else "depth on the background"))
            elif wrong:
                row.update(outcome="wrong label", wrong_as=wrong[0])
            else:
                row.update(outcome="not named")
            if right and right[0].get("score") is not None:
                row["score"] = right[0]["score"]
        else:
            row.update(outcome="hidden")  # in the frustum, nothing of it drawn: occluded (or inside a closed door)
        out.append(row)
    return out


def owl_scores(rows: list[dict], run_dirs: list[Path]) -> None:
    """Fill `owl_score`: OWLv2's best score for the object's label over the boxes on it, any value."""
    from PIL import Image

    from chefbot.explore import owl

    model = owl.OwlV2Model()
    captured: dict = {}
    original = owl.pick_boxes

    def spy(scores, boxes, *rest):
        captured["scores"], captured["boxes"] = scores, boxes
        return original(scores, boxes, *rest)

    owl.pick_boxes = spy
    try:
        by_frame: dict = {}
        for r in rows:
            by_frame.setdefault((r["run"], r["frame"]), []).append(r)
        dirs = {d.name: d for d in run_dirs}
        for (run, frame), group in by_frame.items():
            rgb = np.asarray(Image.open(dirs[run] / "vision" / f"{frame:03d}.png").convert("RGB"))
            model.objects(rgb)
            scores, boxes = captured["scores"], captured["boxes"]
            side = max(rgb.shape[:2])
            centers = boxes[:, :2] * side
            for r in group:
                if "box_w_px" not in r or r["label"] not in model._labels:
                    continue
                q = model._labels.index(r["label"])
                # Boxes centred on the object's visible box (recomputed: rows keep only its size).
                r["owl_score"] = None
                r["_owl_q"] = q
            for r in group:
                if r.get("_owl_q") is None:
                    continue
                q = r.pop("_owl_q")
                x0, y0 = r["_bbox"][:2]
                x1, y1 = r["_bbox"][2:]
                inside = (centers[:, 0] >= x0) & (centers[:, 0] <= x1) & (centers[:, 1] >= y0) & (centers[:, 1] <= y1)
                if inside.any():
                    r["owl_score"] = round(float(scores[inside, q].max()), 3)
                    best = int(scores[inside].max(axis=0).argmax())
                    r["owl_best_label"] = model._labels[best]
    finally:
        owl.pick_boxes = original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--owl-scores", action="store_true")
    parser.add_argument("--analyze", action="store_true", help="only analyze <out>/sightings.csv (no replay)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.analyze:
        print(analyze(args.out / "sightings.csv", args.runs, args.out))
        return 0
    rows = []
    for run in args.runs:
        got = replay(run)
        print(f"{run.name}: {got[0]['frames_replayed'] if got else 0}/{got[0]['frames_logged'] if got else 0} frames, "
              f"{len(got)} sightings", flush=True)  # fmt: skip
        rows += got
    if args.owl_scores:
        owl_scores(rows, args.runs)
    for r in rows:
        r.pop("_bbox", None)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(args.out / "sightings.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out / 'sightings.csv'} ({len(rows)} rows)")
    return 0



# -- analysis of the sighting table -------------------------------------------------------------

PX_BINS = (0, 100, 300, 1000, 3000, 10**9)  # visible pixels
DIST_BINS = (0, 1.0, 1.5, 2.0, 3.0, 99)  # m, camera to object


def _bin(value: float, edges) -> str:
    for lo, hi in zip(edges, edges[1:]):
        if lo <= value < hi:
            return f"{lo}-{hi}" if hi < 10**8 and hi < 99 else f">={lo}"
    return "?"


def per_object(rows: list[dict], found: dict) -> list[dict]:
    """Each true object of each run, with how it ended (found or not) and, if missed, why: the
    furthest stage it reached over all its sightings."""
    by = {}
    for r in rows:
        by.setdefault((r["run"], r["object"]), []).append(r)
    out = []
    for (run, obj), sights in by.items():
        seen = [s for s in sights if int(s["visible_px"]) > 0]
        best = max(seen, key=lambda s: int(s["visible_px"]), default=None)
        outcomes = [s["outcome"] for s in seen]
        rec = {"run": run, "object": obj, "label": sights[0]["label"], "detector": sights[0]["detector"],
               "found": found.get((run, obj), False), "frames_in_view": len(seen),
               "max_px": int(best["visible_px"]) if best else 0,
               "min_distance": min((float(s["distance_m"]) for s in seen), default=None), "size_m": float(sights[0]["size_m"])}  # fmt: skip
        if not seen:
            rec["cause"] = "never visible"
        elif "kept" in outcomes:
            rec["cause"] = "kept (lost later)" if not rec["found"] else "found"
        elif "dropped" in outcomes:
            reasons = [s.get("drop_reason") for s in seen if s["outcome"] == "dropped"]
            rec["cause"] = f"named, then dropped ({max(set(reasons), key=reasons.count)})"
        elif "wrong label" in outcomes:
            rec["cause"] = "wrong label"
            rec["wrong_as"] = next(s.get("wrong_as") for s in seen if s["outcome"] == "wrong label")
        elif rec["max_px"] < 300:
            rec["cause"] = "visible, too few pixels (<300)"
        else:
            rec["cause"] = "visible, not named"
        owl = [float(s["owl_score"]) for s in seen if s.get("owl_score") not in (None, "")]
        rec["owl_best"] = max(owl) if owl else None
        out.append(rec)
    return out


def analyze(csv_path: Path, run_dirs: list[Path], out_dir: Path) -> str:
    """Markdown tables and plots (miss rate vs pixels, error vs distance) from a sighting table."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from chefbot.explore.accuracy import load_run, object_errors

    rows = list(csv.DictReader(open(csv_path)))
    found = {}
    for d in run_dirs:
        for r in object_errors(load_run(d)):
            found[(d.name, r["object"])] = r.get("found_as") is not None
    objs = per_object(rows, found)
    md = []
    for det in ("vision", "owl"):
        sights = [r for r in rows if r["detector"] == det and int(r["visible_px"]) > 0]
        good = lambda r: r["outcome"] == "kept"
        md.append(f"\n### {det}: per sighting (an object visible in a frame), {len(sights)} sightings\n")
        md.append("| visible pixels | sightings | kept | named right (kept or dropped) | wrong label | not named |")
        md.append("|---|---|---|---|---|---|")
        for lo, hi in zip(PX_BINS, PX_BINS[1:]):
            b = [r for r in sights if lo <= int(r["visible_px"]) < hi]
            if b:
                n = len(b)
                md.append(f"| {lo}-{hi if hi < 10**8 else 'more'} | {n} | {100 * sum(map(good, b)) / n:.0f}% | "
                          f"{100 * sum(r['outcome'] in ('kept', 'dropped') for r in b) / n:.0f}% | "
                          f"{100 * sum(r['outcome'] == 'wrong label' for r in b) / n:.0f}% | "
                          f"{100 * sum(r['outcome'] == 'not named' for r in b) / n:.0f}% |")  # fmt: skip
        md.append("\n| distance (m) | sightings | kept | median visible px |")
        md.append("|---|---|---|---|")
        for lo, hi in zip(DIST_BINS, DIST_BINS[1:]):
            b = [r for r in sights if lo <= float(r["distance_m"]) < hi]
            if b:
                md.append(f"| {lo}-{hi if hi < 99 else 'more'} | {len(b)} | {100 * sum(map(good, b)) / len(b):.0f}% | "
                          f"{int(np.median([int(r['visible_px']) for r in b]))} |")  # fmt: skip
        kept = [r for r in sights if good(r)]
        miss = [r for r in sights if not good(r)]
        md.append(f"\nMean distance when kept {np.mean([float(r['distance_m']) for r in kept]):.2f} m "
                  f"({int(np.median([int(r['visible_px']) for r in kept]))} px median), when not "
                  f"{np.mean([float(r['distance_m']) for r in miss]):.2f} m ({int(np.median([int(r['visible_px']) for r in miss]))} px median).")  # fmt: skip
        if kept:
            err = np.array([float(r["error_m"]) for r in kept])
            dist = np.array([float(r["distance_m"]) for r in kept])
            px = np.array([int(r["visible_px"]) for r in kept])
            md.append(f"Position error of kept sightings: median {100 * np.median(err):.1f} cm; correlation with "
                      f"distance {np.corrcoef(dist, err)[0, 1]:+.2f}, with log pixels {np.corrcoef(np.log(px), err)[0, 1]:+.2f}.")  # fmt: skip
        drops = [r.get("drop_reason") for r in sights if r["outcome"] == "dropped"]
        if drops:
            md.append("Named right, then dropped: " + ", ".join(f"{k} {drops.count(k)}" for k in sorted(set(drops))) + ".")
        wrong = [f"{r['label']} as {r['wrong_as']}" for r in sights if r["outcome"] == "wrong label"]
        if wrong:
            md.append("Wrong labels: " + ", ".join(f"{k} ({wrong.count(k)})" for k in sorted(set(wrong), key=wrong.count, reverse=True)) + ".")
        o = [o for o in objs if o["detector"] == det]
        md.append(f"\n**Per object ({len(o)} object appearances over the runs): why each missed one was missed**\n")
        md.append("| cause | objects | labels |")
        md.append("|---|---|---|")
        causes = {}
        for x in o:
            causes.setdefault(x["cause"], []).append(x["label"])
        for cause, labels in sorted(causes.items(), key=lambda kv: -len(kv[1])):
            md.append(f"| {cause} | {len(labels)} | {', '.join(sorted(set(labels)))} |")
    # OWL scores of the right label on visible objects, all frames (both runs' frames).
    scores = [float(r["owl_score"]) for r in rows if r.get("owl_score") not in (None, "")]
    if scores:
        s = np.array(scores)
        md.append(f"\n### OWLv2 score for the right label, on objects in view ({len(s)} sightings, all runs' frames)\n")
        md.append(f"Median {np.median(s):.3f}; share at or above 0.2 (its threshold) {100 * (s >= 0.2).mean():.0f}%, "
                  f"0.15 {100 * (s >= 0.15).mean():.0f}%, 0.10 {100 * (s >= 0.10).mean():.0f}%.")  # fmt: skip
        by_label = {}
        for r in rows:
            if r.get("owl_score") not in (None, ""):
                by_label.setdefault(r["label"], []).append(float(r["owl_score"]))
        md.append("\n| label | sightings | median score | best | most often best name there |")
        md.append("|---|---|---|---|---|")
        for label, v in sorted(by_label.items(), key=lambda kv: np.median(kv[1])):
            names = [r.get("owl_best_label") for r in rows if r["label"] == label and r.get("owl_best_label")]
            md.append(f"| {label} | {len(v)} | {np.median(v):.3f} | {max(v):.3f} | {max(set(names), key=names.count) if names else ''} |")
    # Plots.
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for det, color in (("vision", "#2a6fdb"), ("owl", "#e07b39")):
        sights = [r for r in rows if r["detector"] == det and int(r["visible_px"]) > 0]
        xs, ys = [], []
        for lo, hi in zip(PX_BINS, PX_BINS[1:]):
            b = [r for r in sights if lo <= int(r["visible_px"]) < hi]
            if b:
                xs.append(f"{lo}-{hi if hi < 10**8 else '+'}")
                ys.append(100 * sum(r["outcome"] != "kept" for r in b) / len(b))
        ax.plot(xs, ys, marker="o", color=color, label=det)
    ax.set_xlabel("visible pixels of the object in the frame")
    ax.set_ylabel("miss rate per sighting (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "miss_rate_vs_pixels.png", dpi=130)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for det, color in (("vision", "#2a6fdb"), ("owl", "#e07b39")):
        kept = [r for r in rows if r["detector"] == det and r["outcome"] == "kept"]
        ax.scatter([float(r["distance_m"]) for r in kept], [100 * float(r["error_m"]) for r in kept], s=12, alpha=0.6,
                   color=color, label=det)  # fmt: skip
    ax.set_xlabel("camera to object distance (m)")
    ax.set_ylabel("position error of the sighting (cm)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "error_vs_distance.png", dpi=130)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for det, color in (("vision", "#2a6fdb"), ("owl", "#e07b39")):
        sights = [r for r in rows if r["detector"] == det and int(r["visible_px"]) > 0]
        k = [r for r in sights if r["outcome"] == "kept"]
        m = [r for r in sights if r["outcome"] != "kept"]
        ax.scatter([float(r["distance_m"]) for r in m], [int(r["visible_px"]) for r in m], s=10, marker="x", color=color, alpha=0.5)
        ax.scatter([float(r["distance_m"]) for r in k], [int(r["visible_px"]) for r in k], s=12, color=color, alpha=0.7,
                   label=f"{det}: dot kept, x missed")  # fmt: skip
    ax.set_yscale("log")
    ax.set_xlabel("camera to object distance (m)")
    ax.set_ylabel("visible pixels (log)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pixels_vs_distance.png", dpi=130)
    with open(out_dir / "objects.csv", "w", newline="") as f:
        keys = list(dict.fromkeys(k for o in objs for k in o))
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(objs)
    return "\n".join(md)

if __name__ == "__main__":
    raise SystemExit(main())
