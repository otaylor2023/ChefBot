"""A recorded pass (onboard/collect_stream.py's stream/<pass>) through the laptop's live object map, as the robot
would feed it: each frame to the vision server's detector, into perception/kitchen_map.py, the live page fed as it
goes (scripts/live_view.py), at the recording's pace or as fast as it can. No robot: checks the pipeline, its timings
and the page before a live run, and gives a map to try go_to on.

    VISION_URL=http://127.0.0.1:8777 python -m chefbot.perception.replay_kitchen robot_data/stream/p2_layout3 \\
        --config robot_data/calibration/environments_real_config_2026-11-05.yaml --out runs/replay_p2 [--every 2] [--pace 1]
    python -m chefbot.scripts.live_view --dir runs/live          # watch it

then the map's objects.json / objects_memory.json in --out, and the park stance for each (--goto all).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from chefbot.perception.kitchen_map import KitchenMap
from chefbot.perception.pole_geometry import Pass, load_mount, world_from_camera

REPO = Path(__file__).resolve().parents[2]


class _ReplayEnv:
    """What the live feed asks of an env: localization(), from the recording's mocap track."""

    def __init__(self):
        self.pose = None

    def localization(self):
        from types import SimpleNamespace

        from chefbot.interface.types import Pose2D

        p = None if self.pose is None else Pose2D(*map(float, self.pose))
        return SimpleNamespace(mocap_pose=p, live=p is not None, mocap_age_s=0.0)


def replay(pass_dir, config, out, vision, queries, spec, every=1, pace=0.0, live=None, model="yoloe", name=None,
           log=print) -> KitchenMap:  # fmt: skip
    from chefbot.scripts.live_view import LiveFeed

    p = Pass(pass_dir)
    T_mount = load_mount(config)
    km = KitchenMap(spec, log=log)
    env = _ReplayEnv()
    feed = LiveFeed(live, env, log_dir=out) if live else None
    out = Path(out)
    (out / "vision").mkdir(parents=True, exist_ok=True)
    if feed:
        feed.set(status=f"replaying {p.root.name}")
    t_start, t_rec0 = time.time(), None
    times = {"detect": [], "map": []}
    for k, f in enumerate(p.frames):
        if k % every:
            continue
        pose = p.pose(f)
        if pose is None:
            continue
        if pace and t_rec0 is not None:  # the recording's pace (x pace)
            wait = (float(f["wall"]) - t_rec0) / pace - (time.time() - t_start)
            if wait > 0:
                time.sleep(wait)
        t_rec0 = float(f["wall"]) if t_rec0 is None else t_rec0
        env.pose = pose
        rgb = p.color(f)
        img = out / "vision" / "frame.npy"
        np.save(img, rgb)
        t0 = time.time()
        dets = vision("detect", image=str(img), queries=queries, model=model, threshold=0.15)["detections"]
        t1 = time.time()
        r = km.add_frame(rgb, None, p.camera.color.K, world_from_camera(pose, T_mount), dets, t=float(f["wall"]),
                         xyz=p.camera.aligned_xyz(p.depth(f)))  # fmt: skip
        t2 = time.time()
        times["detect"].append(t1 - t0)
        times["map"].append(t2 - t1)
        if feed:
            feed.frame("pole", rgb, dets)
            feed.timing("detect", t1 - t0, n=len(dets))
            feed.timing("map", t2 - t1, new=len(r["new"]))
            for new in r["new"]:
                feed.event(f"new: {new} (frame {f['n']})")
            feed.objects(km.objects(include_tentative=True))
    merged = km.tidy()
    t_map = time.time() - t_start
    naming = None
    if name:
        if feed:
            feed.set(status=f"naming {len(km.unnamed())} objects ({name})")
        naming = {k: v for k, v in km.name_new(model=name, hint=queries).items() if k != "answers"}
        if feed:
            feed.timing("naming", naming["seconds"], n=naming["named"])
            feed.objects(km.objects(include_tentative=True))
    km.save(out / "objects.json")
    summary = {"pass": p.root.name, "frames": km.n, "objects": len(km.objects()), "merged": merged, "map_seconds": round(t_map, 1),
               "seconds": round(time.time() - t_start, 1), "naming": naming,
               **{f"{k}_ms_median": round(1000 * float(np.median(v)), 1) for k, v in times.items() if v}}  # fmt: skip
    (out / "replay.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary))
    if feed:
        feed.close(f"replayed: {summary['objects']} objects")
    return km


def main(argv=None) -> int:
    from chefbot.perception.detect_frames import load_classes
    from chefbot.runtime import models as M

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pass_dir", type=Path)
    ap.add_argument("--config", type=Path, required=True, help="the environments/real/config.yaml with the pass's pole T")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--map", type=Path, default=REPO / "maps" / "src_kitchen_work")
    ap.add_argument("--vocab", type=Path, default=REPO / "chefbot" / "perception" / "inventory.txt")
    ap.add_argument("--model", default="yoloe")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--pace", type=float, default=0.0, help="1: the recording's pace; 0: as fast as it goes")
    ap.add_argument("--live", type=Path, default=REPO / "runs" / "live")
    ap.add_argument("--goto", nargs="*", default=None, help="the park stance for these objects (or 'all') after")
    ap.add_argument("--name", default=None, help="name the objects after with this VLM (gemini-3.5-flash-lite: GEMINI_API_KEY)")
    a = ap.parse_args(argv)
    url = os.environ.get("VISION_URL")
    if not url:
        raise SystemExit("VISION_URL: the vision server (chefbot/runtime/vision_server.py)")
    km = replay(a.pass_dir.resolve(), a.config.resolve(), a.out.resolve(), M.Vision(url), load_classes(str(a.vocab)),
                a.map / "spec.yaml", every=a.every, pace=a.pace, live=a.live, model=a.model, name=a.name)  # fmt: skip
    if a.goto is not None:
        from chefbot.perception.goto import Floor, park_target
        from chefbot.perception.kitchen_map import counter_box

        floor = Floor.from_spec(a.map / "spec.yaml")
        names = [o["name"] for o in km.objects() if o["status"] == "seen"] if a.goto in ([], ["all"]) else a.goto
        for n in names:
            try:
                o = km.resolve(n)
                pt = park_target(*counter_box(a.map / "spec.yaml", o["counter"])[:3], o["xyz"][:2], floor=floor)
                print(f"  {o['name']:22s} on {o['fixture']:14s} ({o['xyz'][0]:.2f}, {o['xyz'][1]:.2f})  park {pt['side']:5s} "
                      f"stance {pt.get('stance')}" + ("" if pt["within_rule"] else f"  +{pt['reach_extra_m']:.2f} m past the rule"))
            except (LookupError, ValueError) as e:
                print(f"  {n}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
