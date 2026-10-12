"""RobotServer: serves a RobotAPI + Detector over a websocket.

Runs in the same process as the robot layer and the env, so only high-level
calls cross the network. Each connection is handled on its own thread.
"""

from __future__ import annotations

import threading
import traceback

from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, serve

from chefbot.interface.detector import Detector
from chefbot.interface.robot_api import RobotAPI
from chefbot.interface.transport.protocol import (
    ALLOWED_METHODS,
    DETECTOR,
    ROBOT,
    Request,
    Response,
    pack,
    unpack,
)


class RobotServer:
    def __init__(self, robot: RobotAPI, detector: Detector, host: str = "127.0.0.1", port: int = 8765):
        self._targets = {ROBOT: robot, DETECTOR: detector}
        self._server = serve(self._handle, host, port, max_size=None)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._server.socket.getsockname()[1]

    @property
    def url(self) -> str:
        host = self._server.socket.getsockname()[0]
        return f"ws://{host}:{self.port}"

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def start(self) -> None:
        """Serve on a background thread."""
        self._thread = threading.Thread(target=self.serve_forever, name="robot-server", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _handle(self, connection: ServerConnection) -> None:
        try:
            for raw in connection:
                connection.send(pack(self._dispatch(unpack(raw))))
        except ConnectionClosed:
            pass

    def _dispatch(self, request: Request) -> Response:
        if request.method not in ALLOWED_METHODS.get(request.target, ()):
            return Response(
                id=request.id, ok=False, error=f"unknown call {request.target}.{request.method}"
            )
        try:
            result = getattr(self._targets[request.target], request.method)(*request.args)
            return Response(id=request.id, ok=True, result=result)
        except Exception as exc:
            traceback.print_exc()
            return Response(id=request.id, ok=False, error=f"{type(exc).__name__}: {exc}")
