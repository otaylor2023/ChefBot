"""Run a tool trace in a RoboCasa kitchen and record it to a video.

    python -m chefbot.scripts.record_rollout --trace demo --layout 1 --style 1
    python -m chefbot.scripts.record_rollout --trace tour --layout 2 --style 2 --out rollouts
    python -m chefbot.scripts.record_rollout --trace manip      # fridge -> egg -> counter

Each frame shows a third-person view, the robot's shoulder camera (with the
current detections marked), and the occupancy map with the path driven so far.
The tool calls and their results are captioned underneath and saved next to
the video as JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import threading
from collections import Counter
from pathlib import Path
from typing import Callable

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from chefbot.agent.config import AgentConfig
from chefbot.agent.tools.registry import ToolRegistry
from chefbot.agent.world.map import WorldMap
from chefbot.envs.robocasa.config import RoboCasaConfig
from chefbot.envs.robocasa.serve import RoboCasaStack, build_stack
from chefbot.interface.types import MotionStatus, Pose2D, ToolResult
from chefbot.scripts.demo_manip import manip_trace
from chefbot.scripts.demo_nav import TRACE as DEMO_TRACE

VIEW_W, VIEW_H = 640, 480
MAP_SIZE = 480
CAPTION_H = 96
LIST_W = 420  # the semantic map panel
HIGHLIGHT_S = 1.0  # video seconds a new or moved map entry stays highlighted
FONT = cv2.FONT_HERSHEY_SIMPLEX
# PandaOmron base box (length, width), for the map drawing only.
BASE_BOX = (0.70, 0.50)

# tour: the part to look for and align to at each fixture.
TOUR_PARTS = {
    "sink": "faucet",
    "stove": "burner",
    "microwave": "microwave handle",
    "fridge": "fridge handle",
}

Step = tuple[str, dict]


class Recorder:
    """Captures one frame per control tick while the base moves, plus held
    frames between tool calls (sim time does not advance there).

    Frames are requested from the motion thread and the main thread, but all
    rendering happens on one worker thread: MuJoCo's macOS GL context stays
    locked to the thread that last used it.
    """

    def __init__(
        self,
        stack: RoboCasaStack,
        registry: ToolRegistry,
        world_map: WorldMap,
        path: Path,
        speed: int = 1,
        semantic: Callable[[], dict] | None = None,
        remaining: Callable[[], set[str]] | None = None,
    ):
        """`semantic`: () -> {name: {"location", "in"}}, the objects of the map the agent is building,
        drawn on the map and listed in a fourth panel as they appear. `remaining`: () -> names of
        fixtures still to look at (drawn red; the rest green). Both read only the agent's records."""
        self._env = stack.env
        self._semantic = semantic
        self._remaining = remaining
        self._last_objects: dict = {}
        self._last_left: set[str] = set()
        self._changed_at: dict[str, int] = {}  # name -> video frame its entry last appeared or moved
        self._video_frames = 0
        self._registry = registry
        self._map = world_map
        self._grid = stack.robot._planner.grid  # inflated: what the planner sees
        self._raw_grid_path = stack.map_dir
        self._speed = speed
        self._fps = stack.env.config.control_freq
        self._path = path
        self._ticks = 0
        self._trail: list[tuple[float, float]] = []
        self.caption = ("", "")

        self._scene_option = mujoco.MjvOption()
        self._scene_option.geomgroup[:] = 0
        self._scene_option.geomgroup[1] = 1  # visual geometry only, as robosuite renders it
        self._scene_option.sitegroup[:] = 0
        self._camera = self._overview_camera()
        self._map_base, self._map_scale = self._draw_map_base()

        self._jobs: queue.Queue = queue.Queue()
        self._error: BaseException | None = None
        self._worker = threading.Thread(target=self._render_loop, name="recorder", daemon=True)
        self._worker.start()

        # The motion thread ticks the env; hook in behind every tick.
        tick = self._env.wait_for_next_tick

        def tick_and_capture(dt: float) -> None:
            tick(dt)
            self._ticks += 1
            if self._ticks % self._speed == 0:
                self.capture()

        self._env.wait_for_next_tick = tick_and_capture

    def capture(self, repeat: int = 1) -> None:
        """Add the current view to the video `repeat` times. Returns once it is written."""
        done = threading.Event()
        self._jobs.put((repeat, done))
        done.wait()
        if self._error is not None:
            raise RuntimeError("recording failed") from self._error

    def hold(self, seconds: float) -> None:
        self.capture(repeat=max(1, round(seconds * self._fps / self._speed)))

    def close(self) -> None:
        self._jobs.put(None)
        self._worker.join()

    def _render_loop(self) -> None:
        self._renderer = mujoco.Renderer(self._env.model, VIEW_H, VIEW_W)
        writer = imageio.get_writer(self._path, fps=self._fps, codec="libx264", quality=7, macro_block_size=16)
        try:
            while (job := self._jobs.get()) is not None:
                repeat, done = job
                try:
                    if self._error is None:
                        frame = self._frame()
                        for _ in range(repeat):
                            writer.append_data(frame)
                        self._video_frames += repeat
                except BaseException as exc:  # reported to the caller of capture()
                    self._error = exc
                finally:
                    done.set()
        finally:
            writer.close()
            self._renderer.close()

    # -- frame ------------------------------------------------------------

    def _frame(self) -> np.ndarray:
        env = self._env
        with env.lock:
            pose = env.get_state().base_pose
            self._renderer.update_scene(env.data, self._camera, self._scene_option)
            overview = self._renderer.render().copy()
        shoulder = env.get_camera_rgbd(env.config.shoulder_camera)  # takes the lock itself
        self._trail.append((pose.x, pose.y))
        detections = self._registry.tools.detections.all()

        robot_view = shoulder.rgb.copy()
        world_to_cam = np.linalg.inv(shoulder.cam_to_world)
        for det in detections:
            p = world_to_cam @ np.append(det.position_world, 1.0)
            if p[2] < 0.1:
                continue
            u, v, _ = shoulder.intrinsics @ (p[:3] / p[2])
            if 0 <= u < VIEW_W and 0 <= v < VIEW_H:
                cv2.circle(robot_view, (int(u), int(v)), 7, (255, 220, 0), 2, cv2.LINE_AA)
                _text(robot_view, det.id, (int(u) + 10, int(v) + 5), 0.5, (255, 220, 0))
        _text(overview, "overview", (10, 24), 0.6, (255, 255, 255))
        _text(robot_view, "shoulder camera", (10, 24), 0.6, (255, 255, 255))

        panels = [overview, robot_view, self._draw_map(pose, detections)]
        if self._semantic is not None:
            panels.append(self._draw_semantic())
        views = np.hstack(panels)
        caption = np.full((CAPTION_H, views.shape[1], 3), 24, np.uint8)
        _text(caption, self.caption[0], (14, 36), 0.8, (255, 255, 255), outline=False)
        _text(caption, self.caption[1], (14, 76), 0.6, (170, 220, 170), outline=False)
        return np.vstack([views, caption])

    def _overview_camera(self) -> mujoco.MjvCamera:
        """A fixed camera above the open side of the kitchen, looking at the fixtures."""
        fixtures = [e for e in self._map.list() if e.kind == "fixture"]
        points = np.array([e.location for e in fixtures])
        low, high = points.min(axis=0), points.max(axis=0)
        # Fixtures face into the room: look against the direction most of them face.
        facing = Counter((round(e.facing[0]), round(e.facing[1])) for e in fixtures if e.facing)
        fx, fy = facing.most_common(1)[0][0] if facing else (0, -1)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [*(low + high) / 2, 1.2]
        camera.azimuth = math.degrees(math.atan2(-fy, -fx))
        camera.elevation = -45
        camera.distance = 0.55 * float(np.linalg.norm(high - low)) + 2.5
        return camera

    # -- map panel ----------------------------------------------------------

    def _to_px(self, x: float, y: float) -> tuple[int, int]:
        grid, (scale, off_x, off_y, top) = self._grid, self._map_scale
        col = (x - grid.origin[0]) / grid.resolution
        row = (y - grid.origin[1]) / grid.resolution
        return int(off_x + col * scale), int(off_y + (top - row) * scale)  # +y up

    def _draw_map_base(self) -> tuple[np.ndarray, tuple[float, int, int, int]]:
        from chefbot.envs.robocasa.build_map import GRID_FILE
        from chefbot.robot.nav.grid import OccupancyGrid

        raw = OccupancyGrid.load(self._raw_grid_path / GRID_FILE).occupied
        inflated = self._grid.occupied
        cells = np.full((*raw.shape, 3), 245, np.uint8)
        cells[inflated] = (205, 215, 230)  # too close for the base center
        cells[raw] = (70, 70, 80)
        # Crop to the room: the free cells plus a border wide enough to show the fixtures.
        border = round(1.0 / self._grid.resolution)
        free_rows, free_cols = np.nonzero(~inflated)
        row0, row1 = max(0, free_rows.min() - border), min(raw.shape[0], free_rows.max() + border + 1)
        col0, col1 = max(0, free_cols.min() - border), min(raw.shape[1], free_cols.max() + border + 1)
        cells = cells[row0:row1, col0:col1]
        scale = min(MAP_SIZE / cells.shape[1], VIEW_H / cells.shape[0])
        size = (round(cells.shape[1] * scale), round(cells.shape[0] * scale))
        image = cv2.resize(cells[::-1], size, interpolation=cv2.INTER_NEAREST)
        canvas = np.full((VIEW_H, MAP_SIZE, 3), 70, np.uint8)
        canvas[..., 2] = 80
        off_x, off_y = (MAP_SIZE - size[0]) // 2, (VIEW_H - size[1]) // 2
        canvas[off_y : off_y + size[1], off_x : off_x + size[0]] = image
        self._map_scale = (scale, off_x - col0 * scale, off_y, row1)
        for i, entry in enumerate(sorted(self._map.list(), key=lambda e: e.location)):
            u, v = self._to_px(*entry.location)
            cv2.circle(canvas, (u, v), 3, (255, 120, 120), -1, cv2.LINE_AA)
            # Neighbors alternate above / below so their names overlap less.
            _text(canvas, entry.name, (u - 12, v - 7 if i % 2 else v + 15), 0.38, (255, 255, 255))
        return canvas, self._map_scale

    def _draw_map(self, pose: Pose2D, detections) -> np.ndarray:
        canvas = self._map_base.copy()
        if len(self._trail) > 1:
            trail = np.array([self._to_px(x, y) for x, y in self._trail], np.int32)
            cv2.polylines(canvas, [trail], False, (40, 140, 240), 2, cv2.LINE_AA)
        for det in detections:
            cv2.circle(canvas, self._to_px(*det.position_world[:2]), 4, (240, 170, 0), -1, cv2.LINE_AA)
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        half_l, half_w = BASE_BOX[0] / 2, BASE_BOX[1] / 2
        corners = [(half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w)]
        box = np.array(
            [self._to_px(pose.x + c * dx - s * dy, pose.y + s * dx + c * dy) for dx, dy in corners], np.int32
        )
        cv2.polylines(canvas, [box], True, (20, 110, 40), 2, cv2.LINE_AA)
        nose = self._to_px(pose.x + c * half_l, pose.y + s * half_l)
        cv2.arrowedLine(canvas, self._to_px(pose.x, pose.y), nose, (20, 110, 40), 2, cv2.LINE_AA, tipLength=0.4)
        if self._semantic is not None:
            self._draw_map_objects(canvas)
        _text(canvas, "map", (10, 24), 0.6, (255, 255, 255))
        return canvas

    # -- semantic map ---------------------------------------------------------

    def _objects(self) -> dict:
        """The agent's map objects now; notes which are new or moved since the last frame."""
        try:
            objects = self._semantic()
            left = set(self._remaining()) if self._remaining is not None else set()
        except RuntimeError:  # the agent changed its records mid-read: draw the last frame's
            return self._last_objects
        for name, spec in objects.items():
            if self._last_objects.get(name) != spec:
                self._changed_at[name] = self._video_frames
        self._last_objects, self._last_left = objects, left
        return objects

    def _fresh(self, name: str) -> bool:
        at = self._changed_at.get(name)
        return at is not None and self._video_frames - at <= HIGHLIGHT_S * self._fps / self._speed

    def _draw_map_objects(self, canvas: np.ndarray) -> None:
        if self._remaining is not None:
            for entry in self._map.list():
                if entry.kind == "fixture" and entry.label != "burner":
                    color = (235, 70, 70) if entry.name in self._last_left else (60, 200, 90)
                    cv2.circle(canvas, self._to_px(*entry.location), 5, color, -1, cv2.LINE_AA)
        objects = self._objects()
        for i, (name, spec) in enumerate(sorted(objects.items(), key=lambda kv: kv[1]["location"])):
            u, v = self._to_px(*spec["location"])
            if self._fresh(name):
                cv2.circle(canvas, (u, v), 13, (255, 230, 0), 2, cv2.LINE_AA)
            cv2.circle(canvas, (u, v), 6, (60, 220, 90), -1, cv2.LINE_AA)
            cv2.circle(canvas, (u, v), 6, (0, 0, 0), 1, cv2.LINE_AA)
            _text(canvas, name, (u - 14, v + 22 if i % 2 else v + 36), 0.42, (120, 255, 140))

    def _draw_semantic(self) -> np.ndarray:
        """The objects the explored map will hold, as they are added: name, location (cm), fixture."""
        canvas = np.full((VIEW_H, LIST_W, 3), 32, np.uint8)
        objects = self._last_objects
        _text(canvas, f"Semantic map ({len(objects)} objects)", (12, 28), 0.65, (255, 255, 255), outline=False)
        y = 62
        for name, spec in objects.items():
            x, yy = spec["location"]
            color = (255, 230, 0) if self._fresh(name) else (200, 230, 200)
            _text(canvas, name, (14, y), 0.55, color, outline=False)
            _text(canvas, f"({x:.2f}, {yy:.2f})  in: {spec['in']}", (24, y + 20), 0.45, color, outline=False)
            y += 46
        if self._remaining is not None:
            left = sorted(self._last_left)
            note = f"fixtures left to check: {len(left)}" if left else "every fixture checked"
            _text(canvas, note, (12, VIEW_H - 16), 0.5, (235, 120, 120) if left else (120, 230, 140), outline=False)
        return canvas


def _text(image, text, origin, scale, color, outline=True, outline_color=(0, 0, 0)) -> None:
    if outline:
        cv2.putText(image, text, origin, FONT, scale, outline_color, 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, FONT, scale, color, 1, cv2.LINE_AA)


# -- traces ---------------------------------------------------------------------


def demo_trace(run: Callable[[str, dict], ToolResult], world_map: WorldMap) -> None:
    """The scripted trace from demo_nav."""
    for name, args in DEMO_TRACE:
        run(name, args)


def tour_trace(run: Callable[[str, dict], ToolResult], world_map: WorldMap) -> None:
    """Visit each fixture that has a part to reach for: go_to -> detect -> nearby -> align."""
    for place, part in TOUR_PARTS.items():
        if world_map.get(place) is None:
            continue
        if not run("go_to", {"place": place}).success:
            continue
        found = run("detect", {"classes": [part]}).data["detections"]
        run("nearby", {})
        if found:
            run("align", {"id": found[0]["id"]})


def stop_trace(run: Callable[[str, dict], ToolResult], world_map: WorldMap) -> None:
    """Start the longest drive available without waiting, cancel it part-way, then go to the stove."""
    pose = run("get_pose", {}).data["pose"]
    fixtures = [e for e in world_map.list() if e.kind == "fixture"]
    far = max(fixtures, key=lambda e: math.hypot(e.location[0] - pose["x"], e.location[1] - pose["y"]))
    run("go_to", {"place": far.name, "wait": False})
    # Wall-clock seconds; the simulator runs faster than real time.
    run("wait", {"timeout_s": 1.0})
    run("stop", {})
    run("get_status", {})
    run("go_to", {"place": "stove"})
    found = run("detect", {"classes": ["burner"]}).data["detections"]
    if found:
        run("align", {"id": found[0]["id"]})


TRACES = {"demo": demo_trace, "tour": tour_trace, "stop": stop_trace, "manip": manip_trace}


def record(trace: str, layout: int | None, style: int | None, out_dir: Path, speed: int = 1) -> tuple[Path, list[dict]]:
    config = RoboCasaConfig.load(layout=layout, style=style)
    stack = build_stack(config, render=True)
    world_map = WorldMap.load(stack.semantic_map_path)
    registry = ToolRegistry(stack.robot, stack.detector, world_map, AgentConfig.load())
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{trace}_layout{config.layout:03d}_style{config.style:03d}"
    video_path = out_dir / f"{stem}.mp4"
    recorder = Recorder(stack, registry, world_map, video_path, speed)
    log: list[dict] = []

    def run(name: str, args: dict) -> ToolResult:
        call = f"> {name}({', '.join(f'{k}={json.dumps(v)}' for k, v in args.items())})"
        recorder.caption = (call, "...")
        result = registry.dispatch(name, args)
        outcome = "ok" if result.success else f"FAILED: {result.error}"
        summary = f"{outcome}  {result.message}".strip()
        recorder.caption = (call, summary if len(summary) <= 150 else summary[:147] + "...")
        print(f"{call}\n  {summary}", flush=True)
        log.append(
            {"tool": name, "args": args, "success": result.success, "error": result.error,
             "message": result.message, "data": result.data}
        )
        # The motion thread is capturing while a wait=False motion is under way.
        if result.data.get("status") != MotionStatus.MOVING:
            recorder.hold(1.5 if name in ("detect", "nearby") else 0.8)
        return result

    try:
        recorder.caption = (f"layout {config.layout} / style {config.style}: {trace}", "")
        recorder.hold(1.0)
        TRACES[trace](run, world_map)
    finally:
        stack.robot.stop()
        recorder.close()
        stack.env.close()
    (out_dir / f"{stem}.json").write_text(json.dumps(log, indent=2))
    return video_path, log


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a navigation rollout in a RoboCasa kitchen.")
    parser.add_argument("--trace", choices=sorted(TRACES), default="demo")
    parser.add_argument("--layout", type=int, help="kitchen layout (default: config/envs/robocasa.yaml)")
    parser.add_argument("--style", type=int, help="kitchen style (default: config/envs/robocasa.yaml)")
    parser.add_argument("--out", type=Path, default=Path("rollouts"), help="output folder")
    parser.add_argument("--speed", type=int, default=1, help="keep every Nth tick (N x playback speed)")
    args = parser.parse_args()
    video_path, log = record(args.trace, args.layout, args.style, args.out, args.speed)
    failed = [step["tool"] for step in log if not step["success"]]
    print(f"\nwrote {video_path}" + (f"   failed steps: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
