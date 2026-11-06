"""The onboard server: the robot's Env operations over a websocket, for envs/real's RealEnv.

    python -m chefbot.onboard.server [--config chefbot/onboard/config.yaml]

One client at a time. State goes out at `state_hz`; commands are passed to the drivers as they
arrive (robot_io.py); camera frames are grabbed on a worker thread, so a slow camera never holds
up commands. The watchdog: if nothing arrives from the client for `watchdog_s` (the client sends
heartbeats while idle), or the client goes away, the arm holds where it is and the base stops.
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
import traceback

from websockets.exceptions import ConnectionClosed
from websockets.sync.server import ServerConnection, serve

from chefbot.interface.transport import onboard as P
from chefbot.interface.transport.protocol import pack, unpack
from chefbot.onboard.cameras import Cameras
from chefbot.onboard.robot_io import RedisRobot
from chefbot.onboard.stack import CONFIG, build_cameras, build_robot, load_config

class OnboardServer:
    def __init__(self, robot: RedisRobot, cameras: Cameras, config: dict):
        self.robot = robot
        self.cameras = cameras
        self.state_hz = float(config["server"]["state_hz"])
        self.watchdog_s = float(config["server"]["watchdog_s"])
        from chefbot.interface.limits import base_cap, base_stop

        embodiment = config.get("embodiment", "tidybot_fr3")
        cap = base_cap(embodiment)  # told to the client: it checks its speeds against them
        self.limits = {"base_max_vel_xy": cap["vel"][0], "base_max_vel_yaw": cap["vel"][2], **base_stop(embodiment)}
        if cap["accel"] is not None:  # the client sizes its carrot by it (robot/nav/speed.py lookahead_for)
            self.limits["base_max_accel_xy"] = cap["accel"][0]
        self._server = serve(self._handle, config["server"]["host"], int(config["server"]["port"]), max_size=None)
        self._thread: threading.Thread | None = None
        self._client_lock = threading.Lock()
        self._client: ServerConnection | None = None
        self.halts = 0  # how many times the watchdog or a halt stopped the robot
        # A command went to the drivers since the last halt. Disconnects and shutdown halt only then: a client
        # that only watched (state, frames, heartbeats) leaves the drivers' goals as they were.
        self.commanded = False

    @property
    def url(self) -> str:
        host, port = self._server.socket.getsockname()[:2]
        return f"ws://{host}:{port}"

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def start(self) -> None:
        self._thread = threading.Thread(target=self.serve_forever, name="onboard-server", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.cameras.close()

    # -- one client -------------------------------------------------------------

    def _handle(self, connection: ServerConnection) -> None:
        with self._client_lock:
            if self._client is not None:
                connection.send(pack({"type": P.ERROR, "message": "another client is connected"}))
                connection.close()
                return
            self._client = connection
        send_lock = threading.Lock()
        done = threading.Event()

        def send(message: dict) -> None:
            with send_lock:
                connection.send(pack(message))

        frames: queue.Queue = queue.Queue()
        workers = [
            threading.Thread(target=self._stream_state, args=(send, done), name="onboard-state", daemon=True),
            threading.Thread(target=self._serve_frames, args=(frames, send, done), name="onboard-rgbd", daemon=True),
        ]
        moving = False  # commanded since the last halt: silence then means trouble
        try:
            send({"type": P.HELLO, "version": P.VERSION, "cameras": sorted(self.cameras.cameras), "limits": self.limits})
            for worker in workers:
                worker.start()
            while True:
                try:
                    message = unpack(connection.recv(timeout=self.watchdog_s))
                except TimeoutError:
                    if moving:
                        self._halt(send, f"no word from the client for {self.watchdog_s:.2f} s")
                        moving = False
                    continue
                op = message.get("op")
                if op in (P.RGBD, P.CAMERAS_KEEP):
                    frames.put(message)  # on the frame worker: opening a camera takes a second
                elif op in P.COMMANDS:
                    moving = self._command(op, message, send) and op != P.HALT
        except ConnectionClosed:
            pass
        finally:
            done.set()
            frames.put(None)
            self.cameras.kept = set()  # the client's cameras idle out again
            if self.commanded:
                self._halt(None, "the client disconnected")
            with self._client_lock:
                self._client = None

    def _command(self, op: str, m: dict, send) -> bool:
        try:
            if op == P.ARM_JOINTS:
                self.robot.arm_joints(m["q"])
            elif op == P.ARM_POSE:
                self.robot.arm_pose(m["position"], m["orientation"])
            elif op == P.GRIPPER:
                self.robot.gripper(bool(m["closed"]))
            elif op == P.GRIPPER_MOVE:
                self.robot.gripper_move(float(m["width"]), float(m["speed"]))
            elif op == P.GRIPPER_GRASP:
                self.robot.gripper_grasp(float(m["speed"]), float(m["force"]))
            elif op == P.BASE_TARGET_ODOM:
                self.robot.base_target_odom(m["pose"], m["max_vel"])
            elif op == P.BASE_HOLD:
                self.robot.base_hold()
            elif op == P.HALT:
                self._halt(send, m.get("reason", "the client asked"))
            if op != P.HALT:
                self.commanded = True
            return True
        except Exception as exc:  # a bad command must not take the server down
            traceback.print_exc()
            send({"type": P.ERROR, "message": f"{op}: {type(exc).__name__}: {exc}"})
            return False

    def _halt(self, send, reason: str) -> None:
        try:
            self.robot.halt()
        except Exception:
            traceback.print_exc()
        self.commanded = False
        self.halts += 1
        print(f"[onboard] halted: {reason}", flush=True)
        if send is not None:
            try:
                send({"type": P.HALTED, "reason": reason})
            except ConnectionClosed:
                pass

    def _stream_state(self, send, done: threading.Event) -> None:
        period, seq = 1.0 / self.state_hz, 0
        next_t = time.monotonic()
        while not done.is_set():
            try:
                state = self.robot.read_state()
                state.update(type=P.STATE, seq=seq, t=time.time())
                send(state)
                seq += 1
            except ConnectionClosed:
                return
            except Exception:
                traceback.print_exc()
            next_t = max(next_t + period, time.monotonic())
            done.wait(max(0.0, next_t - time.monotonic()))

    def _serve_frames(self, frames: queue.Queue, send, done: threading.Event) -> None:
        while not done.is_set():
            try:
                request = frames.get(timeout=1.0)
            except queue.Empty:
                self.cameras.close_idle()
                continue
            if request is None:
                return
            if request.get("op") == P.CAMERAS_KEEP:
                try:
                    self.cameras.keep(list(request.get("cameras") or []))
                except Exception as exc:
                    send({"type": P.ERROR, "message": f"cameras_keep: {type(exc).__name__}: {exc}"})
                continue
            reply = {"type": P.RGBD, "id": request.get("id")}
            try:
                reply.update(ok=True, **self.cameras.grab(request["camera"]))
            except Exception as exc:
                reply.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            try:
                send(reply)
            except ConnectionClosed:
                return


def build(config: dict, redis_client=None) -> OnboardServer:
    return OnboardServer(build_robot(config, redis_client), build_cameras(config), config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the robot's Env operations to envs/real over a websocket.")
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args()
    server = build(load_config(args.config))
    if not server.robot.wait_gripper_ready():
        print("[onboard] warning: the gripper driver is not ready (still homing?)", flush=True)
    print(f"[onboard] serving on {server.url} (Ctrl-C stops)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server.commanded:
            server.robot.halt()
        server.shutdown()


if __name__ == "__main__":
    main()
