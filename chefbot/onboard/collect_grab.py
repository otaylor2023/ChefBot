"""Robot-side snapshot grabber for data collection (scripts/real_collect.py drives it over ssh). Self-contained: run as a
file with the robot's camera env (pyrealsense2, cv2, redis, numpy); imports nothing from chefbot.

    python collect_grab.py OUT_DIR                 # then, on stdin, one command per line:
        grab STEM                                  # every camera, full resolution, all streams; one JSON line back
        quit

Snapshots only, never a stream to disk: for each grab every camera is opened, warmed up (auto exposure), one frameset
kept, and closed again, so the onboard server can use the cameras between grabs (it opens them on request and closes
them after camera_idle_s). Per camera and grab (OUT_DIR/STEM_<camera>_*):
    color.png            8-bit BGR, the colour sensor's full resolution
    depth.png            16-bit raw depth units (x depth_scale = m), the depth stream's own resolution and viewpoint
    depth_aligned.png    16-bit, the same depth reprojected into the colour image (pixel for pixel with color.png)
    ir1.png / ir2.png    8-bit left / right infrared (D455 both; D405 the one it has)
    imu.json             D455: ~0.3 s of raw accelerometer and gyro samples and their means (gravity -> camera tilt)
    meta.json            device (serial, firmware, USB), every stream's profile, intrinsics + distortion, extrinsics
                         between streams, depth scale, sensor options (exposure, gain, laser, emitter, preset),
                         per-frame timestamps / frame numbers / metadata, and a Redis snapshot taken at the grab:
                         every mocap body (<name>::pos / ::ori / ::tracking_valid), the arm (opensai::sensors::...
                         joint positions etc.), base odometry (hb1::current_pose / _vel / heartbeat)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

CAMERAS = {  # name -> serial, the streams to try (in order), fps
    "pole": {"serial": "049122252230", "plans": [
        {"fps": 15, "streams": [("depth", 1280, 720), ("ir1", 1280, 720), ("ir2", 1280, 720), ("color", 1280, 800)]},
        {"fps": 15, "streams": [("depth", 1280, 720), ("ir1", 1280, 720), ("ir2", 1280, 720), ("color", 1280, 720)]},
        {"fps": 15, "streams": [("depth", 848, 480), ("ir1", 848, 480), ("ir2", 848, 480), ("color", 1280, 800)]},
    ], "imu": False},  # the D455 IMU gave no samples through this path (2026-11-05); the mount is rigid: tilt once, separately
    "wrist": {"serial": "323622271565", "plans": [
        {"fps": 5, "streams": [("depth", 1280, 720), ("ir1", 1280, 720), ("color", 1280, 720)]},
        {"fps": 5, "streams": [("depth", 1280, 720), ("color", 1280, 720)]},
        {"fps": 15, "streams": [("depth", 848, 480), ("color", 848, 480)]},
    ], "imu": False},
}
WARMUP_S = 1.2
REDIS_PATTERNS = ("*::pos", "*::ori", "*::tracking_valid", "opensai::sensors::*", "hb1::current_pose",
                  "hb1::current_vel", "hb1::heartbeat")  # fmt: skip
OPTIONS = ("exposure", "gain", "enable_auto_exposure", "laser_power", "emitter_enabled", "visual_preset", "depth_units",
           "white_balance", "enable_auto_white_balance", "brightness", "contrast", "gamma", "saturation", "sharpness")  # fmt: skip


def _enable(cfg: rs.config, kind: str, w: int, h: int, fps: int) -> None:
    if kind == "depth":
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    elif kind == "color":
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    else:
        cfg.enable_stream(rs.stream.infrared, int(kind[-1]), w, h, rs.format.y8, fps)


def _intr(p) -> dict:
    i = p.as_video_stream_profile().get_intrinsics()
    return {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy,
            "model": str(i.model), "coeffs": list(i.coeffs)}  # fmt: skip


def _extr(a, b) -> dict:
    e = a.get_extrinsics_to(b)
    return {"rotation_colmajor": list(e.rotation), "translation_m": list(e.translation)}


def _frame_meta(f) -> dict:
    out = {"timestamp_ms": f.get_timestamp(), "domain": str(f.get_frame_timestamp_domain()), "frame_number": f.get_frame_number()}
    md = {}
    for name in dir(rs.frame_metadata_value):
        if name.startswith("_"):
            continue
        key = getattr(rs.frame_metadata_value, name)
        try:
            if f.supports_frame_metadata(key):
                md[name] = f.get_frame_metadata(key)
        except Exception:
            pass
    out["metadata"] = md
    return out


def _sensor_options(dev) -> dict:
    out = {}
    for s in dev.query_sensors():
        opts = {}
        for name in OPTIONS:
            opt = getattr(rs.option, name, None)
            if opt is not None and s.supports(opt):
                try:
                    opts[name] = s.get_option(opt)
                except Exception:
                    pass
        out[s.get_info(rs.camera_info.name)] = opts
    return out


def _device(serial: str):
    for d in rs.context().query_devices():
        if d.get_info(rs.camera_info.serial_number) == serial:
            return d
    return None


def grab_camera(name: str, spec: dict, stem: Path, busy_wait_s: float = 20.0) -> dict:
    """grab_camera_once, retried while the device is busy (the onboard server still holding it open: it closes an
    idle camera only while a client is connected, camera_idle_s after its last request)."""
    t_end = time.monotonic() + busy_wait_s
    while True:
        res = grab_camera_once(name, spec, stem)
        busy = not res.get("ok") and any("busy" in e for e in res.get("errors", []))
        if not busy or time.monotonic() > t_end:
            return res
        time.sleep(1.0)


def grab_camera_once(name: str, spec: dict, stem: Path) -> dict:
    dev = _device(spec["serial"])
    if dev is None:
        return {"camera": name, "ok": False, "reason": f"device {spec['serial']} not connected"}
    info = {k: dev.get_info(getattr(rs.camera_info, k)) for k in ("name", "serial_number", "firmware_version",
                                                                  "usb_type_descriptor", "product_line")
            if dev.supports(getattr(rs.camera_info, k))}  # fmt: skip
    errors = []
    for plan in spec["plans"]:
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_device(spec["serial"])
        for kind, w, h in plan["streams"]:
            _enable(cfg, kind, w, h, plan["fps"])
        try:
            prof = pipe.start(cfg)
        except Exception as exc:
            errors.append(f"{plan['streams']}@{plan['fps']}: {exc}")
            continue
        try:
            t_end = time.monotonic() + WARMUP_S
            fs = pipe.wait_for_frames(5000)
            while time.monotonic() < t_end:
                fs = pipe.wait_for_frames(5000)
            t_grab = time.time()
            fs.keep()
            depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
            frames, files, streams = {}, {}, {}
            if fs.get_depth_frame():
                frames["depth"] = fs.get_depth_frame()
            if fs.get_color_frame():
                frames["color"] = fs.get_color_frame()
            for idx in (1, 2):
                f = fs.get_infrared_frame(idx)
                if f:
                    frames[f"ir{idx}"] = f
            for kind, f in frames.items():
                img = np.asanyarray(f.get_data())
                path = Path(f"{stem}_{name}_{kind}.png")
                cv2.imwrite(str(path), img)
                files[kind] = path.name
                streams[kind] = {"intrinsics": _intr(f.get_profile()), "format": str(f.get_profile().format()),
                                 "fps": f.get_profile().fps(), **_frame_meta(f)}  # fmt: skip
            if "depth" in frames and "color" in frames:
                aligned = rs.align(rs.stream.color).process(fs).get_depth_frame()
                path = Path(f"{stem}_{name}_depth_aligned.png")
                cv2.imwrite(str(path), np.asanyarray(aligned.get_data()))
                files["depth_aligned"] = path.name
            extr = {}
            kinds = list(frames)
            for a in kinds:
                for b in kinds:
                    if a != b:
                        extr[f"{a}_to_{b}"] = _extr(frames[a].get_profile(), frames[b].get_profile())
            d = np.asanyarray(frames["depth"].get_data()) if "depth" in frames else None
            meta = {"camera": name, "ok": True, "device": info, "plan": plan, "wall_grab": t_grab,
                    "depth_scale_m": depth_scale, "streams": streams, "extrinsics": extr,
                    "options": _sensor_options(prof.get_device()), "files": files,
                    "depth_valid_frac": None if d is None else float((d > 0).mean()), "errors_before": errors}  # fmt: skip
        finally:
            pipe.stop()
        if spec.get("imu"):
            try:  # never lose the images over the IMU
                meta["imu"] = grab_imu(spec["serial"], Path(f"{stem}_{name}_imu.json"))
                if meta["imu"].get("ok"):
                    files["imu"] = Path(f"{stem}_{name}_imu.json").name
            except Exception as exc:
                meta["imu"] = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        return meta
    return {"camera": name, "ok": False, "device": info, "reason": "no stream plan started", "errors": errors}


def grab_imu(serial: str, path: Path, seconds: float = 0.3) -> dict:
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
    cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
    acc, gyr = [], []
    try:
        prof = pipe.start(cfg)
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}
    try:
        t_end = time.monotonic() + 0.2 + seconds
        t_keep = time.monotonic() + 0.2  # the first samples settle
        while time.monotonic() < t_end:
            fs = pipe.wait_for_frames(5000)
            for f in fs:
                m = f.as_motion_frame()
                if not m or time.monotonic() < t_keep:
                    continue
                v = m.get_motion_data()
                row = [f.get_timestamp(), v.x, v.y, v.z]
                (acc if f.get_profile().stream_type() == rs.stream.accel else gyr).append(row)
        a_prof = next(p for p in prof.get_streams() if p.stream_type() == rs.stream.accel)
        g_prof = next(p for p in prof.get_streams() if p.stream_type() == rs.stream.gyro)
        try:
            a_int = a_prof.as_motion_stream_profile().get_motion_intrinsics()
            motion_intr = {"accel": {"data": [list(r) for r in a_int.data], "noise": list(a_int.noise_variances),
                                     "bias": list(a_int.bias_variances)}}  # fmt: skip
        except Exception:
            motion_intr = None
        out = {"ok": True, "accel_samples": acc, "gyro_samples": gyr, "motion_intrinsics": motion_intr,
               "accel_mean_mps2": np.mean([r[1:] for r in acc], axis=0).tolist() if acc else None,
               "gyro_mean_rps": np.mean([r[1:] for r in gyr], axis=0).tolist() if gyr else None,
               "accel_fps": a_prof.fps(), "gyro_fps": g_prof.fps(),
               "note": "camera axes (x right, y down, z forward); the accel mean is gravity's reaction when still"}  # fmt: skip
    finally:
        pipe.stop()
    path.write_text(json.dumps(out))
    return {k: out[k] for k in ("ok", "accel_mean_mps2", "gyro_mean_rps")} | {"n_accel": len(acc), "n_gyro": len(gyr)}


def redis_snapshot() -> dict:
    try:
        import redis

        r = redis.Redis(decode_responses=True)
        keys = sorted({k for pat in REDIS_PATTERNS for k in r.scan_iter(pat)})
        vals = r.mget(keys) if keys else []
        return {"wall": time.time(), "values": dict(zip(keys, vals))}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    out = Path(sys.argv[1]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"ready": True, "out": str(out)}), flush=True)
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "quit":
            break
        if parts[0] != "grab" or len(parts) != 2:
            print(json.dumps({"ok": False, "reason": f"bad command {line.strip()!r}"}), flush=True)
            continue
        stem = out / parts[1]
        t0 = time.time()
        redis_before = redis_snapshot()
        cams = {}
        for name, spec in CAMERAS.items():
            try:
                cams[name] = grab_camera(name, spec, stem)
            except Exception as exc:
                cams[name] = {"camera": name, "ok": False, "reason": f"{type(exc).__name__}: {exc}"}
            Path(f"{stem}_{name}_meta.json").write_text(json.dumps(cams[name], indent=1, default=str))
        redis_after = redis_snapshot()
        Path(f"{stem}_redis.json").write_text(json.dumps({"before": redis_before, "after": redis_after}, indent=1))
        summary = {"ok": all(c.get("ok") for c in cams.values()), "stem": parts[1], "seconds": round(time.time() - t0, 2),
                   "cameras": {n: {k: c.get(k) for k in ("ok", "reason", "files", "depth_valid_frac", "plan")} for n, c in cams.items()}}  # fmt: skip
        print(json.dumps(summary, default=str), flush=True)


if __name__ == "__main__":
    main()
