"""The memory map built live, while the robot drives the exploration sweep: follows a recording that
chefbot/onboard/collect_stream.py is writing (stream/<pass>/), runs YOLOE on each new pole frame, places what it finds
on the counters (memory_map.MemoryMap.update, the same code as the offline build) and rewrites the map file after every
frame, so `real_nav goto` can use it at any moment. When the recording ends (its summary.json appears) or on Ctrl-C /
SIGTERM it finishes the map (drops one-view entries, merges overlaps) and, with --name, names each object once with a
vision-language model (name_objects.py: ~12 s and < 1 cent for a kitchen with gemini-3.5-flash-lite).

    python -m chefbot.perception.live_map STREAM/sweep_01 --config CAL.yaml --spec maps/src_kitchen/spec.yaml \\
        --classes chefbot/perception/inventory.txt --out maps/src_kitchen/objects.json [--init OLD_MAP.json] \\
        [--device cpu] [--name gemini-3.5-flash-lite --env-file .env]

Real time: frames are taken newest first when it falls behind (it skips, never queues), so the map is at most one
detection behind the camera. Needs the vision env (ultralytics >= 8.4, torch) for YOLOE.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

import numpy as np

from chefbot.perception.memory_map import MemoryMap
from chefbot.perception.pole_geometry import PoleCamera, Track, load_counters, load_mount, world_from_camera

stop = False


def _on_signal(*_):
    global stop
    stop = True


class Follow:
    """A recording that is still being written: new frames as they land, the base's pose from the state so far."""

    def __init__(self, root: Path, cam: str = "pole"):
        self.root, self.cam_dir = Path(root), Path(root) / cam
        self.camera = None
        self._offset = 0
        self._buf = ""

    def ready(self) -> bool:
        if self.camera is None and (self.cam_dir / "meta.json").exists():
            self.camera = PoleCamera.from_meta(json.loads((self.cam_dir / "meta.json").read_text()))
        return self.camera is not None and (self.cam_dir / "frames.jsonl").exists() and (self.root / "state.jsonl").exists()

    def new_frames(self) -> list[dict]:
        with open(self.cam_dir / "frames.jsonl") as f:
            f.seek(self._offset)
            chunk = f.read()
            self._offset = f.tell()
        self._buf += chunk
        *lines, self._buf = self._buf.split("\n")  # keep a half-written last line for next time
        return [json.loads(line) for line in lines if line.strip()]

    def files(self, frame: dict) -> tuple[Path, Path]:
        name = f"{int(frame['n']):06d}.png"
        return self.cam_dir / "color" / name, self.cam_dir / "depth" / name

    def track(self) -> Track:
        return Track.from_state(self.root / "state.jsonl")

    def ended(self) -> bool:
        return (self.root / "summary.json").exists()


def _write(mm: MemoryMap, out: Path, meta: dict) -> None:
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(mm.to_json(meta), indent=1))
    os.replace(tmp, out)  # readers never see a half-written map


def run(a) -> int:
    from PIL import Image

    from chefbot.perception.detect_frames import load_classes, make_model, predict

    # every path absolute first: make_model changes the working directory
    a.env_file = str(Path(a.env_file).resolve()) if a.env_file else None
    a.classes = str(Path(a.classes).resolve()) if a.classes else None
    a.init = str(Path(a.init).resolve()) if a.init else None
    root, out = Path(a.stream).resolve(), Path(a.out).resolve()
    config, spec = Path(a.config).resolve(), Path(a.spec).resolve()
    counters = [c for c in load_counters(spec) if c.height < 1.2]
    mm = MemoryMap.from_json(json.loads(Path(a.init).read_text()), counters) if a.init else MemoryMap(counters)
    mm._pass = root.name
    T_mount = load_mount(config)
    classes = load_classes(a.classes)
    import torch

    torch.set_num_threads(a.threads)  # leave the robot's other cores to the drive loop
    model = make_model(a.model, classes, str(Path(a.weights_dir).resolve()))  # (it changes the working directory)
    meta = {"built_from": [root.name], "init": a.init, "live": True, "classes": a.classes}
    src = Follow(root)
    print(f"waiting for {root} ...", flush=True)
    while not stop and not src.ready():
        time.sleep(0.2)
    done, times, last_log = 0, [], 0
    while not stop:
        frames = src.new_frames()
        if not frames:
            if src.ended():
                break
            time.sleep(0.1)
            continue
        f = frames[-1]  # the newest: skip what we fell behind on
        color_png, depth_png = src.files(f)
        for _ in range(20):  # the recorder writes the PNGs just after (or around) the frame's line
            if color_png.exists() and depth_png.exists():
                break
            time.sleep(0.05)
        else:
            continue
        time.sleep(0.05)  # the state after the frame, for the pose interpolation
        pose = src.track().pose_at(float(f["t_color_ms"]) / 1000.0)
        if pose is None:
            continue
        t0 = time.monotonic()
        try:
            rgb = np.asarray(Image.open(color_png).convert("RGB"))
            depth = np.asarray(Image.open(depth_png))
        except OSError:  # caught mid-write
            continue
        _, dets = predict(model, rgb, a.imgsz, a.conf, a.device)
        xyz = src.camera.aligned_xyz(depth)
        mm.update(dets, xyz, world_from_camera(pose, T_mount), f["n"], float(f["wall"]), src.camera)
        _write(mm, out, meta | {"frames_done": done + 1})
        times.append(time.monotonic() - t0)
        done += 1
        for line in mm.log[last_log:]:
            print("  " + line, flush=True)
        last_log = len(mm.log)
        if done % 10 == 0:
            print(f"[{done} frames, {np.mean(times[-10:]) * 1000:.0f} ms each, skipped {len(frames) - 1} last] "
                  f"{len(mm.entries)} entries", flush=True)  # fmt: skip
    # finish: as the offline build does
    mm.entries = [e for e in mm.entries if e.n_views >= 2 or e.status == "unknown"]
    mm.merge_overlaps()
    mm.rename()
    meta |= {"frames_done": done, "mean_ms": round(1000 * float(np.mean(times)), 1) if times else None, "finished": True}
    _write(mm, out, meta)
    print(f"\n{len(mm.entries)} objects from {done} frames -> {out}", flush=True)
    if a.name:
        from chefbot.perception import name_objects

        argv = [str(out), str(root), "--config", str(config), "--spec", str(spec), "--out", str(out), "--model", a.name]
        argv += ["--hint", a.classes] if a.classes else []
        argv += ["--env-file", a.env_file] if a.env_file else []
        name_objects.main(argv)
    for e in sorted(mm.entries, key=lambda e: (e.counter, e.label)):
        where = f"({e.xyz[0]:.2f}, {e.xyz[1]:.2f})" if e.xyz else "unknown"
        print(f"  {e.name:24s} {e.counter:32s} {where:16s} views {e.n_views:3d}  {e.status}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stream", help="the recording collect_stream.py is writing (stream/<pass>)")
    ap.add_argument("--config", required=True, help="the pole camera's calibration (cameras.pole.T)")
    ap.add_argument("--spec", default="maps/src_kitchen/spec.yaml")
    ap.add_argument("--out", required=True, help="the map file, rewritten after every frame")
    ap.add_argument("--init", help="a map to start from (what changed since)")
    ap.add_argument("--classes", default="chefbot/perception/inventory.txt", help="the vocabulary: what is in this kitchen")
    ap.add_argument("--model", default="yoloe-26s-seg.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--weights-dir", default=os.environ.get("YOLOE_WEIGHTS", str(Path.home() / ".cache" / "yoloe")))
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for the detector (the drive loop shares the CPU)")
    ap.add_argument("--name", default=None, help="name the objects at the end with this VLM (e.g. gemini-3.5-flash-lite)")
    ap.add_argument("--env-file", help="a .env with GEMINI_API_KEY (for --name gemini-*)")
    a = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
