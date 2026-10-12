"""RemoteRobot / RemoteDetector: the RobotAPI and Detector over a websocket.

Each calling thread gets its own connection, so a blocking `wait()` on one
thread never holds up `stop_base()` on another.
"""

from __future__ import annotations

import itertools
import threading
from typing import Any

import numpy as np
from websockets.sync.client import ClientConnection, connect

from chefbot.interface.detector import Detector
from chefbot.interface.robot_api import RobotAPI
from chefbot.interface.transport.protocol import DETECTOR, ROBOT, Request, Response, pack, unpack
from chefbot.interface.types import Detection, MotionResult, Pose2D


class RemoteError(RuntimeError):
    """The server raised while handling a call."""


class _Channel:
    def __init__(self, url: str, target: str, open_timeout_s: float = 10.0):
        self._url = url
        self._target = target
        self._open_timeout_s = open_timeout_s
        self._local = threading.local()
        self._ids = itertools.count()
        self._lock = threading.Lock()
        self._connections: list[ClientConnection] = []

    def _connection(self) -> ClientConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            # Entered here and closed in close(): the connection outlives this call.
            conn = connect(self._url, max_size=None, open_timeout=self._open_timeout_s).__enter__()
            self._local.conn = conn
            with self._lock:
                self._connections.append(conn)
        return conn

    def call(self, method: str, *args: Any) -> Any:
        conn = self._connection()
        request = Request(id=next(self._ids), target=self._target, method=method, args=list(args))
        conn.send(pack(request))
        response: Response = unpack(conn.recv())
        if response.id != request.id:
            raise RemoteError(f"response id {response.id} does not match request {request.id}")
        if not response.ok:
            raise RemoteError(response.error or "remote call failed")
        return response.result

    def close(self) -> None:
        with self._lock:
            connections, self._connections = self._connections, []
        for conn in connections:
            conn.close()
        self._local = threading.local()


class RemoteRobot(RobotAPI):
    def __init__(self, url: str):
        self._channel = _Channel(url, ROBOT)

    def go_to_pose(self, goal: Pose2D) -> None:
        self._channel.call("go_to_pose", goal)

    def align_to_point(self, target: np.ndarray) -> None:
        self._channel.call("align_to_point", np.asarray(target, dtype=float))

    def wait(self, timeout_s: float | None = None) -> MotionResult:
        return self._channel.call("wait", timeout_s)

    def get_status(self) -> MotionResult:
        return self._channel.call("get_status")

    def stop_base(self) -> None:
        self._channel.call("stop_base")

    def get_base_pose(self) -> Pose2D:
        return self._channel.call("get_base_pose")

    def close(self) -> None:
        self._channel.close()


class RemoteDetector(Detector):
    def __init__(self, url: str):
        self._channel = _Channel(url, DETECTOR)

    def detect(self, classes: list[str]) -> list[Detection]:
        return self._channel.call("detect", list(classes))

    def close(self) -> None:
        self._channel.close()
