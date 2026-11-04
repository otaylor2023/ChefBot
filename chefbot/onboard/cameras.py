"""The RealSense cameras, one aligned RGB-D frame on request.

A camera opens on its first request and closes after `idle_s` without one, so nothing streams
next to the arm's real-time loop while no one is looking. Images go out as they are; the camera's
pose is the other side's to work out (from its mount, the base pose and the arm).
"""

from __future__ import annotations

import threading
import time

import numpy as np


class Camera:
    def __init__(self, name: str, serial: str, width: int = 640, height: int = 480, fps: int = 15, warmup_frames: int = 5):
        self.name, self.serial = name, str(serial)
        self.width, self.height, self.fps, self.warmup_frames = width, height, fps, warmup_frames
        self._pipeline = None
        self._align = None
        self._depth_scale = 0.001
        self._intrinsics: np.ndarray | None = None
        self.last_used = 0.0

    @property
    def open(self) -> bool:
        return self._pipeline is not None

    def grab(self) -> dict:
        import pyrealsense2 as rs  # only on the robot computer

        if self._pipeline is None:
            pipeline, config = rs.pipeline(), rs.config()
            config.enable_device(self.serial)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            profile = pipeline.start(config)
            self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            i = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self._intrinsics = np.array([[i.fx, 0.0, i.ppx], [0.0, i.fy, i.ppy], [0.0, 0.0, 1.0]])
            self._align = rs.align(rs.stream.color)
            self._pipeline = pipeline
            for _ in range(self.warmup_frames):  # auto exposure settles
                pipeline.wait_for_frames()
        frames = self._align.process(self._pipeline.wait_for_frames())
        self.last_used = time.monotonic()
        return {
            "rgb": np.asanyarray(frames.get_color_frame().get_data()).copy(),
            "depth": np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * self._depth_scale,
            "intrinsics": self._intrinsics.copy(),
        }

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None


class Cameras:
    """The robot's cameras by name, from the onboard config's `cameras:` list."""

    def __init__(self, specs: list[dict], idle_s: float = 10.0):
        self.cameras = {s["name"]: Camera(s["name"], s["serial"], s.get("width", 640), s.get("height", 480), s.get("fps", 15)) for s in specs}
        self.idle_s = idle_s
        self._lock = threading.Lock()
        self.kept: set[str] = set()  # open for the client however long idle (a skill's cameras)

    def grab(self, name: str) -> dict:
        if name not in self.cameras:
            raise KeyError(f"no camera {name!r} (have: {sorted(self.cameras)})")
        with self._lock:
            return self.cameras[name].grab()

    def keep(self, names) -> None:
        """Hold these open (opening them now) and let the others idle out; [] holds none."""
        unknown = [n for n in names if n not in self.cameras]
        if unknown:
            raise KeyError(f"no camera {unknown} (have: {sorted(self.cameras)})")
        self.kept = set(names)
        for name in self.kept:
            with self._lock:
                if not self.cameras[name].open:
                    self.cameras[name].grab()  # starts it (and its auto exposure settles)

    def close_idle(self) -> None:
        with self._lock:
            for name, camera in self.cameras.items():
                if name not in self.kept and camera.open and time.monotonic() - camera.last_used > self.idle_s:
                    camera.close()

    def close(self) -> None:
        with self._lock:
            for camera in self.cameras.values():
                camera.close()
