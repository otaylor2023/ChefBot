"""The real robot's cameras, delivered as the fridge lab's simulator delivers them: 640x360 RGB,
metric depth (0 where there is none; None for the RGB-only wrist camera), intrinsics at that size,
and the lens model ("pinhole", or "fisheye" with OpenCV's k1..k4).

Capture follows MessyMem's homer/envs/utils/cameras.py (ZED SDK settings, USB lookup by serial): each
camera grabs in its own thread and `frame()` returns the latest. Where a camera sits is the host's
business (calibration in robot_real.yaml): ZEDs in the arm base frame, the wrist camera in the flange's.
"""

from __future__ import annotations

import glob
import threading
import time

import cv2
import numpy as np

from experiments.fridge_lab import fisheye

W, H = 640, 360


class ZedCamera:
    """A ZED 2: left image and depth at HD720, scaled to 640x360."""

    model = "pinhole"

    def __init__(self, serial: str, exposure: int | None = None, gain: int | None = None, depth_mode: str = "NEURAL_PLUS", fps: int = 30):
        import pyzed.sl as sl

        self.sl, self.zed = sl, sl.Camera()
        p = sl.InitParameters()
        p.set_from_serial_number(int(serial))
        p.camera_resolution = sl.RESOLUTION.HD720
        p.camera_fps = fps
        p.depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
        p.coordinate_units = sl.UNIT.METER
        p.depth_minimum_distance = 0.1
        p.depth_maximum_distance = 10.0
        status = self.zed.open(p)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED {serial}: {status}")
        auto = exposure is None and gain is None
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1 if auto else 0)
        if exposure is not None:
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, int(exposure))
        if gain is not None:
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, int(gain))
        info = self.zed.get_camera_information().camera_configuration
        c, s = info.calibration_parameters.left_cam, W / info.resolution.width
        self.K = np.array([[c.fx * s, 0, c.cx * s], [0, c.fy * s, c.cy * s], [0, 0, 1.0]])
        self.D = None
        self._latest, self._stop = None, False
        self._first = threading.Event()
        threading.Thread(target=self._grab, daemon=True).start()

    def _grab(self) -> None:
        sl = self.sl
        rt, img, dep = sl.RuntimeParameters(), sl.Mat(), sl.Mat()
        while not self._stop:
            if self.zed.grab(rt) != sl.ERROR_CODE.SUCCESS:
                continue
            self.zed.retrieve_image(img, sl.VIEW.LEFT)
            self.zed.retrieve_measure(dep, sl.MEASURE.DEPTH)
            rgb = cv2.resize(cv2.cvtColor(img.get_data(), cv2.COLOR_BGRA2RGB), (W, H), interpolation=cv2.INTER_AREA)
            d = dep.get_data().copy()
            d[~np.isfinite(d)] = 0.0
            self._latest = (rgb, cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32), time.time())
            self._first.set()

    def frame(self) -> tuple[np.ndarray, np.ndarray | None, float]:
        """(rgb, depth m, capture time)"""
        if not self._first.wait(10.0):
            raise RuntimeError("ZED: no frame in 10 s")
        return self._latest

    def close(self) -> None:
        self._stop = True
        time.sleep(0.1)
        self.zed.close()


class UsbFisheyeCamera:
    """The Arducam wrist camera (V4L2, found by serial), scaled to 640x360. RGB only."""

    model = "fisheye"

    def __init__(self, serial: str, width: int = 1920, height: int = 1080, K=None, D=None):
        path = next((p for p in glob.glob("/dev/v4l/by-id/*") if serial in p and "video-index0" in p), None)
        if path is None:
            raise RuntimeError(f"no USB camera with serial {serial} under /dev/v4l/by-id")
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"could not open {path}")
        self.cap = cap
        self.K = np.asarray(K, dtype=float) if K is not None else fisheye.K_of(fisheye.nominal_f())
        self.D = list(D) if D is not None else list(fisheye.D)
        self.calibrated = K is not None
        self._latest, self._stop = None, False
        self._first = threading.Event()
        threading.Thread(target=self._grab, daemon=True).start()

    def _grab(self) -> None:
        while not self._stop:
            ok, bgr = self.cap.read()
            if ok and bgr is not None:
                self._latest = (cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA), None, time.time())
                self._first.set()

    def frame(self) -> tuple[np.ndarray, None, float]:
        if not self._first.wait(10.0):
            raise RuntimeError("wrist camera: no frame in 10 s")
        return self._latest

    def close(self) -> None:
        self._stop = True
        time.sleep(0.1)
        self.cap.release()


class FakeCamera:
    """No hardware: a gray image and a flat wall 1.5 m ahead (pinhole), or a gray fisheye image."""

    def __init__(self, model: str = "pinhole"):
        self.model = model
        self.K = fisheye.K_of(fisheye.nominal_f()) if model == "fisheye" else np.array([[266.0, 0, W / 2], [0, 266.0, H / 2], [0, 0, 1.0]])
        self.D = list(fisheye.D) if model == "fisheye" else None
        self.calibrated = True

    def frame(self):
        rgb = np.full((H, W, 3), 128, np.uint8)
        return rgb, None if self.model == "fisheye" else np.full((H, W), 1.5, np.float32), time.time()

    def close(self) -> None:
        pass


def make_camera(name: str, c: dict, dry_run: bool):
    if dry_run:
        return FakeCamera("fisheye" if c["type"] == "usb" else "pinhole")
    if c["type"] == "zed":
        return ZedCamera(c["serial"], c.get("exposure"), c.get("gain"), c.get("depth_mode", "NEURAL_PLUS"))
    if c["type"] == "usb":
        return UsbFisheyeCamera(c["serial"], c.get("width", 1920), c.get("height", 1080), c.get("K"), c.get("D"))
    raise ValueError(f"camera {name}: unknown type {c['type']!r}")
