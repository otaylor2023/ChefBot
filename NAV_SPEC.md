# ChefBot Navigation Tools — Implementation Spec (v0.4, sim)

## 1. Goal

Build the navigation tool layer the ChefBot cooking agent (an LLM calling tools) uses to move a mobile manipulator around a kitchen. Development happens in simulation, but **the agent must be environment-agnostic**: the same agent code runs against RoboCasa now, Isaac later, and the real robot eventually, with only the env backend swapped.

Exploration/mapping is **out of scope**: maps are generated from simulator ground truth and loaded from files.

Navigation is two-stage:
- **Coarse — `go_to(place)`**: room scale, from the map. Gets the robot near a fixture with it in view. ±10 cm / ±10° is fine.
- **Fine — `detect` → `nearby` → `align(id)`**: last-meter scale, from what the robot currently sees. Positions the base so a specific detected item is within arm reach. ±2 cm target.

## 2. Architecture: agent / robot / env

The agent decides, the robot plans and controls, the env actuates.

```
┌──────────────── AGENT (env-agnostic) ──────────────────────┐
│ tool registry · tools · map (semantic) · detections ·      │
│ nearby view                                                │
└───────────────┬────────────────────────────────────────────┘
                │ RobotAPI: go_to_pose, align_to_point, wait, get_status,
                │ stop_base, get_base_pose  +  Detector: detect
                │ in-process, or RemoteRobot over websocket  ← only network hop
┌───────────────┴──────────────────────── ROBOT (env-agnostic) ┐
│ occupancy grid · A* · standoff-pose selection · FR3          │
│ kinematics / reachability · base controller · motion thread  │
└───────────────┬──────────────────────────────────────────────┘
                │ Env API: set_base_velocity, get_state, wait_for_next_tick(dt), ...
                │ always in-process
     ┌──────────┴────────────┬────────────────────┐
     ▼                       ▼                    ▼
envs/robocasa          envs/isaac (later)    envs/real (later)
physics, GT detector,  same shape            mocap + TidyBot + FR3 drivers,
map builder                                  real detector
```

Rules (enforced by tests, §12):
1. **The agent knows no geometry of the robot.** No footprint, no arm kinematics, no occupancy grid, no paths. It chooses *goals* (poses, object IDs); the robot layer decides how to get there.
2. **Controllers, planning, and kinematics live in the robot layer (`chefbot/robot/`), never in an env.** Envs only accept low-level commands and report state.
3. **The robot layer runs in the same process as the env**, so control loops never cross the network. If the agent is remote, only high-level calls go over the wire.
4. **`chefbot/agent/` and `chefbot/robot/` import no simulator or hardware library** (robosuite, robocasa, isaaclab, omni.*, hardware drivers) and never import `chefbot/envs/`. Exception: `robot/kinematics/` may import `mujoco`/`mink` to load the FR3 MJCF (a robot description, not the simulator).
5. **Everything crossing a boundary is a plain serializable type** from `chefbot/interface/`. No sim handles, no env-specific names.
6. **Env-specific knowledge stays in the env** (e.g. the RoboCasa-name ↔ ChefBot-label table).
7. **Two agent↔robot transports, same API:** in-process, and `RemoteRobot` over websocket + msgpack-numpy (openpi-style). Integration tests run over both.

## 3. Environment (v0.1 backend)

- **Simulator:** RoboCasa (robosuite + MuJoCo). Must run on macOS (Apple Silicon); no NVIDIA dependency in agent, robot, or RoboCasa backend.
- **Robot:** RoboCasa's default mobile Franka (stand-in for TidyBot++ + FR3). Assume holonomic — **verify RoboCasa's base action space first and flag if it differs.**
- **Python:** 3.10+, type hints throughout. Separate dependency sets via pyproject extras (`chefbot[agent]`, `chefbot[robot]`, `chefbot[robocasa]`) so the agent installs without any sim.
- **Localization:** sim ground-truth base pose (stands in for mocap).

Note for later: VoLo (NVIDIA, 2026) found open-vocab vision models and sim-trained VLA checkpoints transfer poorly in MuJoCo-rendered scenes. Irrelevant for v0.1 (ground-truth detector, no VLA), but real perception and VLA testing will likely need an Isaac backend or the real robot — which is why this split exists.

## 4. Repo layout

```
chefbot/
  interface/                  # shared; numpy + msgpack only
    types.py                  # Pose2D, Detection, MotionResult, EnvState, RGBD, ToolResult
    robot_api.py              # RobotAPI ABC (§5.1)
    detector.py               # Detector ABC (§5.2)
    env.py                    # Env ABC (§5.3)
    transport/
      protocol.py             # message schema, msgpack-numpy codec
      client.py               # RemoteRobot(RobotAPI), RemoteDetector(Detector)
      server.py               # RobotServer: serves a Robot + Detector over websocket
  robot/                      # our robot software stack; env-agnostic; runs next to the env
    robot.py                  # Robot(RobotAPI), built on any Env
    motion.py                 # motion thread, preemption, status
    nav/
      grid.py                 # OccupancyGrid (§6.1)
      astar.py                # AStarPlanner (§6.2)
      goal.py                 # goal snapping + standoff-pose selection (§6.3)
      base_controller.py      # path following -> base velocity commands (§6.4)
    kinematics/
      fr3.py                  # FR3 MJCF (MuJoCo Menagerie) + IK via mink
      reachability.py         # ReachabilityChecker (§6.5)
  agent/                      # env-agnostic; no robot geometry
    world/
      map.py                  # semantic map: fixtures + rough locations (§7.1)
      detections.py           # latest detection results (§7.2)
      nearby.py               # nearby view: visible + not-visible items (§7.3)
    tools/
      nav.py                  # go_to, detect, nearby, align, wait, get_status, stop, get_pose (§8)
      registry.py             # JSON tool schemas + dispatch (§9)
  envs/
    robocasa/
      env.py                  # RoboCasaEnv(Env)
      detector.py             # GroundTruthDetector(Detector) + name table
      build_map.py            # RoboCasa scene -> semantic_map.yaml + grid.npy/grid.yaml
      serve.py                # RobotServer(Robot(RoboCasaEnv()), GroundTruthDetector)
    isaac/                    # placeholder (README only)
    real/                     # placeholder (README only)
  config/
    agent.yaml                # approach_distance, nearby_radius, max_align_distance, map_match_radius, refine_radius, default_vocab
    robot.yaml                # footprint, control rate, gains, limits, tolerances, standoff range, IK params
    envs/robocasa.yaml        # scene/layout, cameras, physics timestep, server host/port
  scripts/
    demo_nav.py               # scripted trace; --env robocasa | --remote ws://host:port
  tests/
    agent/                    # FakeRobot + FakeDetector; no sim installed
    robot/                    # FakeEnv; no sim installed
    transport/
    envs/robocasa/
    test_boundaries.py
```

## 5. Interfaces

### 5.1 RobotAPI (what the agent calls)

```python
class RobotAPI(ABC):
    def go_to_pose(self, goal: Pose2D) -> None: ...          # non-blocking; robot plans + drives
    def align_to_point(self, target: np.ndarray) -> None: ... # non-blocking; robot picks a reachable
                                                              # standoff pose around target (world xyz), plans + drives
    def wait(self, timeout_s: float | None = None) -> MotionResult: ...  # None = until motion ends
    def get_status(self) -> MotionResult: ...                # non-blocking snapshot
    def stop_base(self) -> None: ...                         # preempts any motion, callable anytime
    def get_base_pose(self) -> Pose2D: ...                   # world frame; x, y (m), yaw (rad)
```

- `MotionResult.status`: `idle | moving | success | stopped | blocked | no_path | unreachable | timeout | error`, plus `final_pose`, `message`.
- Starting a new motion while one is running preempts the old one (old result: `stopped`).
- `wait` with a timeout that expires returns `status="moving"`; motion continues.
- Yaw is radians in the interface; degrees only at the LLM boundary.

### 5.2 Detector

```python
class Detector(ABC):
    def detect(self, classes: list[str]) -> list[Detection]: ...
# Detection: label, position_world (3,), confidence
```

- Looks through the shoulder camera from the robot's current pose.
- Takes ChefBot labels only; each env maps them internally.
- **v0.1: `envs/robocasa/GroundTruthDetector`** — returns world positions of sim objects whose label matches `classes` **and that are visible**: inside the shoulder camera frustum and not occluded (raycast). Without the visibility check, nothing is ever "not seen" and recovery paths go untested.
- **Later:** `OpenVocabDetector` (open-vocab detection + depth → world point).

### 5.3 Env (what the robot layer calls; low-level only)

```python
class Env(ABC):
    def get_state(self) -> EnvState: ...   # base pose, base velocity, base_in_contact, time
    def set_base_velocity(self, vx: float, vy: float, vyaw: float, frame: str = "world") -> None: ...
    def set_arm_joint_targets(self, q: np.ndarray) -> None: ...   # unused by nav v0.1; present for later
    def get_camera_rgbd(self, cam: str) -> RGBD: ...
    def wait_for_next_tick(self, dt: float) -> None: ...  # sim: step physics by dt; real: likely no-op (see below)
```

- Envs contain no controllers or planning. `wait_for_next_tick(dt)` is the only notion of time on the interface, which is what lets identical robot-layer code run in sim and on hardware.
- Real env (later): likely a no-op, **provided something else paces the loop** — e.g. `get_state()` blocks until a new mocap sample arrives. If nothing blocks, the control loop spins at CPU speed, so the real env must pace it in one of these two places.

## 6. Robot layer (`chefbot/robot/`)

### 6.1 OccupancyGrid
- Loaded from `grid.npy` + `grid.yaml` (resolution, origin). `is_free`, `world_to_cell`, `cell_to_world`.
- Inflated by footprint radius (`robot.yaml`) + safety margin. All planning uses the inflated grid.

### 6.2 AStarPlanner
- 8-connected A*, Euclidean heuristic, line-of-sight shortcutting. Raises `NoPath`. Final waypoint carries goal yaw; intermediate yaws interpolate.

### 6.3 Goal handling
- **`go_to_pose`:** if the goal is in collision, snap to the nearest free cell within `snap_radius` (keep goal yaw); else `no_path`. The agent's approach poses are rough by design, so this is expected.
- **`align_to_point`:** sample candidate base poses on rings around the target (`[standoff_min, standoff_max]`, N angles), yaw facing the target. Keep collision-free AND reachable (§6.5). Pick lowest A* cost from current pose. None valid → `unreachable`.

### 6.4 Motion thread + BaseController
- Each motion runs on a dedicated thread. Each tick: check preempt flag → `env.get_state()` → P/PD on position, yaw interpolated → velocity/accel limits → `env.set_base_velocity` → `env.wait_for_next_tick(dt)`.
- Ends with `success` (within tolerance), `stopped` (preempted), `blocked` (contact or path deviation > threshold), or `timeout`. Velocity is always zeroed on exit.
- Safety stops are independent of the agent.

### 6.5 ReachabilityChecker
- `is_reachable(base_pose, target_world) -> bool` using the FR3 MJCF (MuJoCo Menagerie) and the arm mount transform from `robot.yaml`. Position-only IK for v0.1.
- **TCP offset:** no gripper model in v0.1. IK targets the flange such that a point `tcp_offset` along the flange z-axis reaches the target. Set `tcp_offset` to the Robotiq 2F-85 length (flange to fingertip, ~0.17 m — verify against the Menagerie 2F-85 model) so standoff poses match the real gripper. Uses its own MjData; never touches sim state.

## 7. Agent world state

Three pieces, named for what they are:

### 7.1 Map (persistent)

Env-agnostic, no sim names, no parts, no geometry beyond rough locations. Contract between exploration (future) and the cooking agent (now). Parts (burners, handles) are **not** in the map; they're found by detection.

```yaml
frame: world
fixtures:
  stove:     { location: [3.4, 1.0], facing: [0.0, 1.0] }   # facing optional, toward open floor
  fridge:    { location: [0.6, 2.1], facing: [1.0, 0.0] }
  counter_1: { location: [2.0, 1.0], facing: [0.0, 1.0] }
  counter_2: { location: [2.0, 3.2], facing: [0.0, -1.0] }
objects:
  egg_carton: { location: [0.6, 2.1], in: fridge }
```

API: `get(name)`, `list()`, `in_range(pose, radius)`. A map entry's **label** is its name minus any `_n` suffix (`counter_2` → `counter`).

### 7.2 Detections (scratch buffer)

The results of the latest `detect` call, in the world frame. Each entry: `id`, `label`, `position_world`, `confidence`, `seen_from` (base pose), `timestamp`, and optional `map_name` (§7.3).

- **Overwritten on every `detect`** (v0.1 simplification).
- IDs: requested-class detections get `label_n` (`burner_1`, `pan_1`). Detections matched to a map entry take the map entry's name (`stove`, `counter_2`).
- `align`'s refine step updates only its target entry.

### 7.3 Nearby (view, never stored)

What the agent sees via `nearby()`. Built from the map, the detections, and the **current** pose:

- **Visible (alignable):** detections within `nearby_radius` of the current pose, including detected map entries.
- **Nearby, not visible:** map entries within `nearby_radius` that the last `detect` did not find, for whatever reason (behind the robot, occluded, missed). Not alignable; `go_to` first.
- Sorted by distance; **distances are not shown** to the agent. Range handling belongs to `align`/`go_to`.
- Re-filtered against the current pose on every call, so items drop off as the robot drives away. Visible / not-visible status is from the last `detect`; if the robot turns afterward, the agent calls `detect` again.

**Map matching:** during `detect`, each map entry in range is queried by its label. A detection with that label is assigned to the nearest unassigned map entry within `map_match_radius`; unmatched detections of map labels are dropped (they're the map's to report). Sim: trivial via ground truth. Real: use the map label as the detector prompt.

## 8. Agent tools

All return `ToolResult(success, error, message, data)`; poses in degrees in `data`. Motion tools take `wait: bool = True`; with `wait=False` they return immediately with `status: "moving"`.

### `go_to(place: str, wait=True)`
1. Look up `place` in the map → else `unknown_place` (message lists known places).
2. Goal = `location + approach_distance · facing` (if no `facing`, the direction from `location` toward the current base pose), yaw facing `location`.
3. `robot.go_to_pose(goal)`; if `wait`, `robot.wait()`.
- Errors: `unknown_place`, `no_path`, `blocked`, `timeout`, `stopped`.

### `detect(classes: list[str] | None = None)`
1. `classes` defaults to `default_vocab` (agent.yaml; fixed kitchen list).
2. Query = `classes` ∪ labels of map entries within `nearby_radius`.
3. `detector.detect(query)` → match map entries (§7.3) → overwrite detections.
4. **Return only results for the requested `classes`** (ids + labels, sorted by distance, no distances). Map entries are checked silently and appear only in `nearby()`.
- Nothing found for the requested classes → `success` with an empty list and a message saying so.

### `nearby()`
- Returns the nearby view (§7.3): visible items with ids, then nearby-not-visible map entries. Does not run detection.

### `align(id: str, wait=True)`
1. Look up `id` in the nearby view.
   - Not present → `unknown_id`; message includes the current nearby list (or says to call `detect` if empty).
   - Nearby but not visible → `not_visible`; message: `go_to("<id>") first`.
2. Distance guard (backstop): xy distance base→item > `max_align_distance` → `too_far`, message: use `go_to` first.
3. `robot.align_to_point(position)`; if `wait`, `robot.wait()`.
4. On success: re-detect the item's label; take the nearest detection within `refine_radius` of the stored position and update that entry only. If none found, still `success` with `data.refined = false`.
- Errors: `unknown_id`, `not_visible`, `too_far`, `unreachable`, `no_path`, `blocked`, `timeout`, `stopped`.

### `wait(timeout_s: float | None = None)`
- Waits for the current motion to end (or until timeout → `status: "moving"`). Returns the motion's result as a ToolResult. No motion → `idle`.

### `get_status()`
- Non-blocking motion status.

### `stop()`
- `robot.stop_base()`. Valid anytime, including during motion started with `wait=False`.

### `get_pose()`
- Current base pose (degrees). Never fails.

## 9. Tool registry

`registry.py`: OpenAI-style function-calling schemas for all tools + `dispatch(name, args) -> ToolResult`. Constructed with `(robot: RobotAPI, detector: Detector, world_map, config)`; never knows which env it's using. Descriptions are written for an LLM and state the workflow: **`go_to` a fixture → `detect` what you need → `nearby` to see everything around you → `align` a visible id**; `align` only works on visible items from `nearby()`, and `stop` works mid-motion when motion was started with `wait=False`.

## 10. Config

- `agent.yaml`: `approach_distance`, `nearby_radius`, `max_align_distance`, `map_match_radius`, `refine_radius`, `default_vocab`.
- `robot.yaml`: footprint radius, safety margin, arm mount transform, `tcp_offset`, control rate, gains, velocity/accel limits, pos/yaw tolerances, deviation threshold, timeouts, `snap_radius`, standoff range + ring sampling, IK tolerance.
- `envs/robocasa.yaml`: scene/layout, cameras, physics timestep, server host/port.

## 11. Map building (env-side)

`envs/robocasa/build_map.py`: from a RoboCasa scene, (a) project collision geoms in a height band (e.g. 0.05–1.8 m) onto the floor → `grid.npy` + `grid.yaml` (robot layer), and (b) write fixture/object rough locations and facing → `semantic_map.yaml` (agent). The RoboCasa ↔ ChefBot name table is shared with `GroundTruthDetector`.

## 12. Acceptance criteria

**Boundaries** (`tests/test_boundaries.py`):
- No module under `agent/`, `robot/`, or `interface/` imports robosuite, robocasa, isaaclab, omni, or `chefbot.envs`; `mujoco`/`mink` only under `robot/kinematics/`.
- `agent/` imports nothing from `chefbot.robot` (agent knows only RobotAPI).
- Envs don't import `chefbot.robot`.
- Agent and robot test suites pass without robosuite/robocasa installed.

**Robot layer** (FakeEnv: kinematic integrator; optional fake obstacle sets `base_in_contact`):
- Grid round trip; inflation correct. A* solves a maze; `NoPath` when enclosed; shortcut path collision-free.
- Controller reaches goals within tolerance and respects limits; `blocked` on contact; `timeout`; velocity zero on every exit.
- `stop_base` during motion → `stopped` within one control tick; new motion preempts old.
- `wait()` returns on completion; `wait(timeout)` returns `moving` early.
- Goal snapping works; `align_to_point` returns `unreachable` for a target surrounded by obstacles.
- Reachability: point in front of arm reachable; 2 m away not.

**Transport:** every message type round-trips through the codec; `stop_base` works over `RemoteRobot` while a remote `wait()` is in progress.

**Agent** (FakeRobot + FakeDetector):
- Each tool's error paths, including: `unknown_id` message lists the nearby view; `not_visible` for undetected map entries; `too_far` backstop.
- `detect` returns only requested classes, while matched map entries appear as visible in `nearby()` and unmatched in-range map entries as not visible.
- Map matching assigns detections to the nearest map entry within `map_match_radius`.
- `nearby()` drops items as FakeRobot's pose moves away; `detect` overwrites detections; `align` refinement updates only its entry.

**Integration** (RoboCasa, ≥2 layouts; run in-process **and** over `RemoteRobot`):
- `go_to` every fixture from 3 random free starts.
- `go_to("stove") → detect(["burner"]) → nearby() → align(<a burner id>)`; same for fridge handle and microwave handle. Final reach verified by IK.
- Standing at the stove facing it, a map entry behind the robot shows as nearby-not-visible, and `align` on it returns `not_visible`.
- `go_to("fridge", wait=False) → stop()` halts the base.
- `scripts/demo_nav.py` runs `go_to("stove") → detect(["burner", "pan"]) → nearby() → align("burner_1") → go_to("fridge") → detect(["fridge handle"]) → align("fridge handle_1")` and prints each ToolResult, with both `--env robocasa` and `--remote`.
- No base collisions with non-floor geoms.

## 13. Future work (flagged, not v0.1)

- **Stable detection IDs** across `detect` calls (associate by label + position) instead of overwriting. Removes the risk of the agent reusing an old ID that now points at a different object.
- **Part/fixture aggregation** (burner_1 belongs to stove) without precomputing parts in the map.
- **Visibility refresh** when the robot turns after `detect` (currently the agent re-calls `detect`).
- **Re-detecting the right object** in `align`'s refine step — e.g. pass a crop / appearance features from the original detection instead of label + nearest position.
- **Annotated detection image** returned to the LLM (set-of-marks style IDs drawn on the frame).
- **Verified poses written back to the map** after successful aligns (learning-from-doing, à la MessyMem).
- **Idle sim stepping:** sim time only advances while the robot layer calls `wait_for_next_tick` (i.e. during motion). Once anything changes on its own (timers, heating), the robot layer must keep stepping the sim while idle.
- **Gripper model** (Robotiq 2F-85 from MuJoCo Menagerie) when manipulation tools arrive; replaces `tcp_offset`.
- VLM-generated default vocabulary; open-vocab detector; Isaac and real backends; dynamic obstacles; `go_to_point` (VLM map pointing).

## 14. Open questions

- RoboCasa base action space (holonomic? velocity or position control?).
- Shoulder camera: RoboCasa's existing robot cameras, or add one matching our real mast mount?
- Kitchen layouts to standardize on for tests.
- Transport: websocket + msgpack-numpy proposed; confirm before the Isaac backend.
