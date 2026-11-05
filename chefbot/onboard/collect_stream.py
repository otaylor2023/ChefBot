"""Robot-side recorder for data collection while the base drives (real_nav park at teleop speed): streams both
RealSenses and saves every frame, with the robot's state sampled alongside on the same clock. Self-contained: run as a
file with the robot's camera env (pyrealsense2, cv2, redis, numpy); imports nothing from chefbot.

    python collect_stream.py OUT_DIR [--seconds 0]      # SIGTERM / Ctrl-C stops it cleanly

OUT_DIR/
    <camera>/color/<n>.png, <camera>/depth/<n>.png    every frame, lossless (depth 16-bit raw units x depth_scale = m)
    <camera>/frames.jsonl     per frame: n, hardware frame number, timestamp (ms, host clock: global time on),
                              wall time received, blur score (variance of the Laplacian, 320 px wide), depth valid fraction
    <camera>/meta.json        device, USB link, stream profiles, intrinsics + distortion, depth->color extrinsics,
                              depth scale, sensor options
    state.jsonl               each change of the robot's state, sampled at ~200 Hz: mocap bodies (pos / ori /
                              tracking_valid), base odometry and velocity, the base command sequence, arm joints
    summary.json              counts, rates, drops, at the end

Depth stays in the depth camera's own view (not aligned): the meta has what is needed to align it offline.
"""

from __future__ import annotations

import json
import queue
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

CAMERAS = {
    "pole": {"serial": "049122252230", "color": (1280, 800), "depth": (1280, 720), "fps": 15},
    "wrist": {"serial": "323622271565", "color": (640, 480), "depth": (640, 480), "fps": 15},  # USB 2.1 link (2026-11-05)
}
STATE_KEYS = ["chefbot-tidy::pos", "chefbot-tidy::ori", "chefbot-tidy::tracking_valid", "Chef-botcounter-top::pos",
              "Chef-botcounter-top::ori", "Chef-botcounter-top::tracking_valid", "hb1::current_pose", "hb1::current_vel",
              "hb1::cmd_seq", "hb1::desired_pose", "hb1::heartbeat", "opensai::sensors::FrankaRobot::joint_positions"]  # fmt: skip
PNG = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # fast; still lossless

stop = threading.Event()


def _intr(p) -> dict:
    i = p.as_video_stream_profile().get_intrinsics()
    return {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy,
            "model": str(i.model), "coeffs": list(i.coeffs)}  # fmt: skip


def _options(dev) -> dict:
    out = {}
    for s in dev.query_sensors():
        opts = {}
        for name in ("exposure", "gain", "enable_auto_exposure", "laser_power", "emitter_enabled", "visual_preset",
                     "depth_units", "global_time_enabled", "auto_exposure_priority"):  # fmt: skip
            opt = getattr(rs.option, name, None)
            if opt is not None and s.supports(opt):
                try:
                    opts[name] = s.get_option(opt)
                except Exception:
                    pass
        out[s.get_info(rs.camera_info.name)] = opts
    return out


def camera_loop(name: str, spec: dict, out: Path, writes: queue.Queue, stats: dict) -> None:
    (out / name / "color").mkdir(parents=True, exist_ok=True)
    (out / name / "depth").mkdir(parents=True, exist_ok=True)
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(spec["serial"])
    cfg.enable_stream(rs.stream.color, *spec["color"], rs.format.bgr8, spec["fps"])
    cfg.enable_stream(rs.stream.depth, *spec["depth"], rs.format.z16, spec["fps"])
    try:
        prof = pipe.start(cfg)
    except Exception as exc:
        stats[name] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        print(json.dumps({"camera": name, "error": stats[name]["reason"]}), flush=True)
        return
    dev = prof.get_device()
    for s in dev.query_sensors():
        if s.supports(rs.option.global_time_enabled):
            s.set_option(rs.option.global_time_enabled, 1)
        if s.supports(rs.option.auto_exposure_priority):  # keep the frame rate (shorter exposure: less blur)
            try:
                s.set_option(rs.option.auto_exposure_priority, 0)
            except Exception:
                pass
    cs, ds = prof.get_stream(rs.stream.color), prof.get_stream(rs.stream.depth)
    meta = {"camera": name, "serial": spec["serial"], "usb": dev.get_info(rs.camera_info.usb_type_descriptor),
            "firmware": dev.get_info(rs.camera_info.firmware_version), "fps": spec["fps"],
            "color": _intr(cs), "depth": _intr(ds), "depth_scale_m": dev.first_depth_sensor().get_depth_scale(),
            "depth_to_color": {"rotation_colmajor": list(ds.get_extrinsics_to(cs).rotation),
                               "translation_m": list(ds.get_extrinsics_to(cs).translation)},
            "options": _options(dev), "started_wall": time.time()}  # fmt: skip
    (out / name / "meta.json").write_text(json.dumps(meta, indent=1))
    log = (out / name / "frames.jsonl").open("w")
    n, st = 0, {"ok": True, "frames": 0, "dropped_hw": 0, "last_hw": None}
    stats[name] = st
    print(json.dumps({"camera": name, "streaming": True, "usb": meta["usb"]}), flush=True)
    try:
        while not stop.is_set():
            try:
                fs = pipe.wait_for_frames(2000)
            except RuntimeError:
                continue
            c, d = fs.get_color_frame(), fs.get_depth_frame()
            if not c or not d:
                continue
            color = np.asanyarray(c.get_data()).copy()
            depth = np.asanyarray(d.get_data()).copy()
            small = cv2.cvtColor(cv2.resize(color, (320, int(320 * color.shape[0] / color.shape[1]))), cv2.COLOR_BGR2GRAY)
            hw = c.get_frame_number()
            if st["last_hw"] is not None and hw > st["last_hw"] + 1:
                st["dropped_hw"] += hw - st["last_hw"] - 1
            st["last_hw"] = hw
            row = {"n": n, "hw_frame": hw, "t_color_ms": c.get_timestamp(), "t_depth_ms": d.get_timestamp(),
                   "domain": str(c.get_frame_timestamp_domain()).split(".")[-1], "wall": time.time(),
                   "blur": float(cv2.Laplacian(small, cv2.CV_64F).var()), "depth_valid": float((depth > 0).mean())}  # fmt: skip
            try:
                row["exposure_us"] = c.get_frame_metadata(rs.frame_metadata_value.actual_exposure)
            except Exception:
                pass
            log.write(json.dumps(row) + "\n")
            writes.put((out / name / "color" / f"{n:06d}.png", color))
            writes.put((out / name / "depth" / f"{n:06d}.png", depth))
            n += 1
            st["frames"] = n
    finally:
        log.close()
        pipe.stop()


def writer(writes: queue.Queue, stats: dict) -> None:
    while True:
        item = writes.get()
        if item is None:
            return
        path, img = item
        cv2.imwrite(str(path), img, PNG)
        stats["written"] = stats.get("written", 0) + 1


def state_loop(out: Path, stats: dict, hz: float = 200.0) -> None:
    import redis

    r = redis.Redis(decode_responses=True)
    last, n = {}, 0
    with (out / "state.jsonl").open("w") as f:
        while not stop.is_set():
            t = time.time()
            vals = r.mget(STATE_KEYS)
            changed = {k: v for k, v in zip(STATE_KEYS, vals) if last.get(k) != v}
            if changed:
                f.write(json.dumps({"wall": t, **changed}) + "\n")
                last.update(changed)
                n += 1
            time.sleep(max(0.0, 1.0 / hz - (time.time() - t)))
    stats["state_rows"] = n


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    seconds = float(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else 0.0
    out.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stats: dict = {"started_wall": time.time()}
    writes: queue.Queue = queue.Queue(maxsize=600)  # ~20 s of both cameras; a full queue slows capture, never drops silently
    writers = [threading.Thread(target=writer, args=(writes, stats), daemon=True) for _ in range(6)]
    for w in writers:
        w.start()
    threads = [threading.Thread(target=state_loop, args=(out, stats), daemon=True)]
    threads += [threading.Thread(target=camera_loop, args=(n, s, out, writes, stats), daemon=True) for n, s in CAMERAS.items()]
    for t in threads:
        t.start()
    t0 = time.time()
    while not stop.is_set() and (seconds <= 0 or time.time() - t0 < seconds):
        time.sleep(0.2)
    stop.set()
    for t in threads:
        t.join(5)
    for _ in writers:
        writes.put(None)
    for w in writers:
        w.join(30)
    stats["stopped_wall"] = time.time()
    dur = stats["stopped_wall"] - stats["started_wall"]
    for name in CAMERAS:
        if isinstance(stats.get(name), dict) and stats[name].get("frames"):
            stats[name]["fps"] = round(stats[name]["frames"] / dur, 2)
    (out / "summary.json").write_text(json.dumps(stats, indent=1, default=str))
    print(json.dumps({"done": True, **{k: v for k, v in stats.items() if k in CAMERAS or k in ("written", "state_rows")}},
                     default=str), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
