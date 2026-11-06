"""A RealSense frame for calibration: color, depth aligned to it (m) and the color camera's intrinsics.

    from chefbot.onboard.calibration.camera import Camera
    with Camera("323622271565") as cam:      # the wrist D405
        f = cam.frame()                      # {"bgr", "depth", "K", "D"}
"""

from __future__ import annotations

import numpy as np

WRIST_D405 = "323622271565"
POLE_D455 = "049122252230"


class Camera:
    def __init__(self, serial: str, width: int = 1280, height: int = 720, fps: int = 15):
        import pyrealsense2 as rs

        self.rs, self.serial = rs, serial
        self.pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        try:
            prof = self.pipe.start(cfg)
        except RuntimeError:  # on USB 2 the D405 has 1280x720 only up to 15 fps, and less over a busy hub
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 5)
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, 5)
            prof = self.pipe.start(cfg)
        self.depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
        i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1.0]])
        self.D = np.array(i.coeffs[:5], dtype=float)
        self.align = rs.align(rs.stream.color)
        for _ in range(15):  # auto exposure settles
            self.pipe.wait_for_frames()

    def frame(self) -> dict:
        f = self.align.process(self.pipe.wait_for_frames())
        bgr = np.asanyarray(f.get_color_frame().get_data()).copy()
        depth = np.asanyarray(f.get_depth_frame().get_data()).astype(np.float32) * self.depth_scale
        return {"bgr": bgr, "depth": depth, "K": self.K, "D": self.D}

    def close(self) -> None:
        self.pipe.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
