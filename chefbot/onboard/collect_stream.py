"""Robot-side recorder for data collection while the base drives (real_nav park at teleop speed): streams both
RealSenses and saves the frames taken while the base moves (A to B; --all: every frame), with the robot's state
sampled alongside on the same clock. Self-contained: run as a
file with the robot's camera env (pyrealsense2, cv2, redis, numpy); imports nothing from chefbot.

    python collect_stream.py OUT_DIR [--seconds 0] [--all] [--hz 3]   # SIGTERM / Ctrl-C stops it cleanly
    (--hz: frames written per second while moving, 0 = all 15)

OUT_DIR/
    <camera>/color/<n>.png, depth/<n>.png, ir1/ (ir2/) <n>.png   lossless (depth 16-bit raw units x depth_scale = m;
                              infrared 8-bit, the stereo pair at the depth stream's resolution)
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
    "pole": {"serial": "049122252230", "color": (1280, 800), "depth": (1280, 720), "fps": 15, "ir": [1, 2]},
    "wrist": {"serial": "323622271565", "color": (640, 480), "depth": (640, 480), "fps": 15, "ir": [1]},  # USB 2.1 link (2026-11-05)
}
BODIES = ["chefbot-tidy", "Chef-botcounter-top", "kitchen_island_countertop", "1-corner-countertop",
          "2-corner-connected-1-and-fridge", "far-counter"]  # every Motive body we map with (2026-11-05)
STATE_KEYS = [f"{b}::{k}" for b in BODIES for k in ("pos", "ori", "tracking_valid")] + [
    "hb1::current_pose", "hb1::current_vel", "hb1::cmd_seq", "hb1::desired_pose", "hb1::desired_vel", "hb1::heartbeat",
    "hb1::max_vel_scale", "hb1::stop",
    "opensai::sensors::FrankaRobot::joint_positions", "opensai::sensors::FrankaRobot::joint_velocities",
    "opensai::sensors::FrankaRobot::joint_torques"]  # fmt: skip
PNG = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # fast; still lossless

stop = threading.Event()
# Motion gating (the default): frames are written only while the base moves (odometry speed or turn rate over the
# thresholds), plus MOTION_PAD_S before and after, so an A-to-B drive is captured and the time the base stands still
# (planning, waiting, parked) is not. --all writes every frame.
MOVE_MPS, MOVE_RADPS, MOTION_PAD_S = 0.02, 0.05, 0.5
SAVE_ALL = False
SAVE_HZ = 3.0  # frames written per second while moving (--hz; 0: every frame): consecutive 15 fps frames add little
last_saved: dict = {}  # camera -> wall time of the last frame written
last_motion = [0.0]  # wall time the base was last seen moving
pending: dict = {}  # camera -> frames held for the pre-motion pad


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
    for i in spec.get("ir", []):
        (out / name / f"ir{i}").mkdir(parents=True, exist_ok=True)
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(spec["serial"])
    cfg.enable_stream(rs.stream.color, *spec["color"], rs.format.bgr8, spec["fps"])
    cfg.enable_stream(rs.stream.depth, *spec["depth"], rs.format.z16, spec["fps"])
    for i in spec.get("ir", []):  # infrared: the stereo pair's own images, at the depth stream's resolution
        cfg.enable_stream(rs.stream.infrared, i, *spec["depth"], rs.format.y8, spec["fps"])
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
            "ir": {f"ir{i}": _intr(prof.get_stream(rs.stream.infrared, i)) for i in spec.get("ir", [])},
            "ir_from_depth": {f"ir{i}": {"rotation_colmajor": list(ds.get_extrinsics_to(prof.get_stream(rs.stream.infrared, i)).rotation),
                                         "translation_m": list(ds.get_extrinsics_to(prof.get_stream(rs.stream.infrared, i)).translation)}
                              for i in spec.get("ir", [])},
            "depth_to_color": {"rotation_colmajor": list(ds.get_extrinsics_to(cs).rotation),
                               "translation_m": list(ds.get_extrinsics_to(cs).translation)},
            "options": _options(dev), "started_wall": time.time()}  # fmt: skip
    (out / name / "meta.json").write_text(json.dumps(meta, indent=1))
    log = (out / name / "frames.jsonl").open("w")
    n, st = 0, {"ok": True, "frames": 0, "dropped_hw": 0, "last_hw": None}
    stats[name] = st
    print(json.dumps({"camera": name, "streaming": True, "usb": meta["usb"]}), flush=True)
    events = (out / name / "events.jsonl").open("a")
    st["restarts"] = 0
    try:
        while not stop.is_set():
            try:
                fs = pipe.wait_for_frames(2000)
            except RuntimeError as exc:  # no frame in 2 s: the stream stalled (a USB error); restart it, and say so
                ev = {"wall": time.time(), "event": "stall", "error": str(exc), "frames_so_far": n}
                try:
                    pipe.stop()
                except Exception:
                    pass
                time.sleep(1.0)
                try:
                    pipe.start(cfg)
                    ev["restarted"] = True
                    st["restarts"] += 1
                except Exception as exc2:
                    ev["restarted"] = False
                    ev["restart_error"] = str(exc2)
                events.write(json.dumps(ev) + "\n")
                events.flush()
                print(json.dumps({"camera": name, **ev}), flush=True)
                continue
            c, d = fs.get_color_frame(), fs.get_depth_frame()
            if not c or not d:
                continue
            color = np.asanyarray(c.get_data()).copy()
            depth = np.asanyarray(d.get_data()).copy()
            irs = {}
            for i in spec.get("ir", []):
                f = fs.get_infrared_frame(i)
                if f:
                    irs[i] = np.asanyarray(f.get_data()).copy()
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
            now = time.time()
            moving = SAVE_ALL or now - last_motion[0] <= MOTION_PAD_S
            buf = pending.setdefault(name, [])
            if not moving:  # hold the last MOTION_PAD_S of frames, in case motion starts
                buf.append((row, color, depth, irs))
                while buf and now - buf[0][0]["wall"] > MOTION_PAD_S:
                    buf.pop(0)
                    st["skipped_still"] = st.get("skipped_still", 0) + 1
                continue
            batch = buf + [(row, color, depth, irs)]
            if SAVE_HZ > 0:  # thin to SAVE_HZ (the pre-motion pad too)
                kept = []
                for item in batch:
                    if item[0]["wall"] - last_saved.get(name, 0.0) >= 1.0 / SAVE_HZ - 1e-3:
                        kept.append(item)
                        last_saved[name] = item[0]["wall"]
                    else:
                        st["thinned"] = st.get("thinned", 0) + 1
                batch = kept
            for brow, bcolor, bdepth, birs in batch:
                brow["n"] = n
                brow["ir"] = sorted(birs)
                log.write(json.dumps(brow) + "\n")
                writes.put((out / name / "color" / f"{n:06d}.png", bcolor))
                writes.put((out / name / "depth" / f"{n:06d}.png", bdepth))
                for i, img in birs.items():
                    writes.put((out / name / f"ir{i}" / f"{n:06d}.png", img))
                n += 1
            buf.clear()
            st["frames"] = n
    finally:
        log.close()
        events.close()
        try:
            pipe.stop()
        except Exception:
            pass


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
            vel = dict(zip(STATE_KEYS, vals)).get("hb1::current_vel")
            if vel:
                try:
                    vx, vy, w = json.loads(vel)
                    if (vx * vx + vy * vy) ** 0.5 > MOVE_MPS or abs(w) > MOVE_RADPS:
                        last_motion[0] = t
                except (ValueError, TypeError):
                    pass
            if changed:
                f.write(json.dumps({"wall": t, **changed}) + "\n")
                last.update(changed)
                n += 1
            time.sleep(max(0.0, 1.0 / hz - (time.time() - t)))
    stats["state_rows"] = n


def main() -> None:
    global SAVE_ALL, SAVE_HZ
    SAVE_ALL = "--all" in sys.argv
    if "--hz" in sys.argv:
        SAVE_HZ = float(sys.argv[sys.argv.index("--hz") + 1])
    out = Path(sys.argv[1]).expanduser()
    seconds = float(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else 0.0
    out.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    stats: dict = {"started_wall": time.time()}
    writes: queue.Queue = queue.Queue(maxsize=600)  # ~20 s of both cameras; a full queue slows capture, never drops silently
    writers = [threading.Thread(target=writer, args=(writes, stats), daemon=True) for _ in range(8)]
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
