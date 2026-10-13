# ChefBot Manipulation Tools — Implementation Spec (v0.1, sim)

Builds on NAV_SPEC.md (v0.4). Same three layers (agent / robot / env), same rules: the agent decides, the robot plans and controls, the env actuates. Read NAV_SPEC §2 first.

## 1. Goal

Give the cooking agent real arm-based manipulation before any VLA exists:
- **Pick and place loose objects** with the real arm and gripper (classical primitives).
- **Open and close fixtures** (fridge, microwave, cabinets, drawers) via **scripted** sim calls for now, behind the same tool interface a real implementation will use later.

The only things faked in sim are **perception inputs** (where the object is, which grasp to use). Motion, IK, and gripper control are real and env-agnostic, so they carry over to Isaac and the real robot.

Design reference: VoLo (NVIDIA, 2026) `grasp`/`place` primitives (segmentation → grasp proposal → IK → execute). Our sim versions replace the learned perception models with ground truth, because VoLo found GroundingDINO/SAM/Molmo fail on MuJoCo renders.

## 2. Scope

**In v0.1**
- Robotiq 2F-85 gripper on the sim robot (replacing the Panda hand).
- Arm control: joint-space and straight-line Cartesian moves, IK, contact-aware abort.
- `pick(id)`, `place(id)`, `open(id)`, `close(id)` agent tools, plus held-object state.
- Ground-truth `GraspProposer` and `PlacementProposer` for RoboCasa.
- A RoboCasa scene with graspable objects.

**Out of v0.1** (see §13): real door/drawer opening with the arm, obstacle-aware arm motion planning, learned grasp proposals, VLA tools, placing inside containers/shelves, pouring, tool use.

## 3. Gripper change

- Configure the sim robot with robosuite's `Robotiq85Gripper`. **Verify RoboCasa's mobile robot accepts the gripper override** and flag if not.
- Remove `tcp_offset` from NAV_SPEC's reachability: the TCP becomes the gripper's grip site (fingertip center) read from the robot description. Reachability and grasp IK both target this TCP frame.
- Note any obvious mismatch between robosuite's 2F-85 model and the real one (stroke, finger length) in the PR; we may swap to the MuJoCo Menagerie 2F-85 later.

## 4. Interface changes

### 4.1 Env (low-level; additions to NAV_SPEC §5.3)

```python
class Env(ABC):
    # existing: get_state, set_base_velocity, get_camera_rgbd, wait_for_next_tick
    def set_arm_joint_targets(self, q: np.ndarray) -> None: ...      # 7-DoF position targets
    def set_gripper(self, closed: bool) -> None: ...                  # binary for v0.1
```

`EnvState` gains: `arm_q` (7,), `arm_qd` (7,), `gripper_width` (m), `gripper_closed_cmd` (bool), `arm_in_contact` (bool: unexpected arm/hand contact, excluding fingers–target), `ee_wrench` (optional, 6,).

Envs still contain no controllers: they apply the latest targets and step.

### 4.2 Perception proposers (new ABCs in `interface/`)

Like `Detector`, these are provided by the env in sim and by real perception later. The robot layer receives them by injection; it never imports an env.

```python
class GraspProposer(ABC):
    def propose(self, target: Detection) -> list[Grasp]: ...
# Grasp: tcp_pose_world (4x4), pregrasp_offset (m, along approach axis), width_hint (m), score

class PlacementProposer(ABC):
    def propose(self, surface: Detection, held: HeldObject) -> list[Placement]: ...
# Placement: tcp_pose_world (4x4) at release, preplace_offset (m), score
```

### 4.3 FixtureActuator (new ABC; scripted in v0.1)

```python
class FixtureActuator(ABC):
    def set_state(self, fixture: Detection, open: bool) -> ActuationResult: ...
    def get_state(self, fixture: Detection) -> str: ...   # "open" | "closed" | "unknown"
# ActuationResult: success, scripted (bool), message
```

RoboCasa implementation calls the fixture helpers (`open_door()`, `close_door()`, `is_open()`, drawer equivalents). `scripted=True` always in v0.1.

### 4.4 RobotAPI (additions to NAV_SPEC §5.1)

```python
class RobotAPI(ABC):
    # existing: go_to_pose, align_to_point, wait, get_status, get_base_pose
    def stop(self) -> None: ...                         # replaces stop_base: halts base AND arm
    def pick(self, target: Detection) -> None: ...      # non-blocking
    def place(self, surface: Detection) -> None: ...    # non-blocking
    def actuate_fixture(self, fixture: Detection, open: bool) -> None: ...  # non-blocking
    def get_held(self) -> HeldObject | None: ...
```

All motion calls share the single motion thread from NAV_SPEC §6.4: one active motion at a time, a new call preempts, `stop()` preempts anything, `wait()` / `get_status()` work for all of them.

`MotionResult.status` gains: `grasp_failed`, `no_grasp`, `no_placement`, `not_holding`, `already_holding`, `collision`.

## 5. Robot layer additions (`chefbot/robot/`)

```
robot/
  arm/
    arm_controller.py     # joint-space interpolation + Cartesian line moves -> set_arm_joint_targets
    ik.py                 # whole-arm IK via mink on FR3 MJCF (shared with reachability)
    gripper.py            # open/close + grasp verification from gripper_width
  skills/
    pick.py
    place.py
    actuate.py
  safety.py               # contact abort, base/arm interlock
```

### 5.1 ArmController
- **Joint moves:** interpolate from current `arm_q` to a goal `q` with joint velocity/accel limits (`robot.yaml`); send targets each tick.
- **Cartesian line moves:** straight-line TCP path (position lerp + orientation slerp), sampled at `cart_step` (~1 cm), IK solved per sample seeded from the previous solution; abort with `no_ik` if any sample fails or a joint jump exceeds `max_joint_jump`.
- Every tick checks `arm_in_contact` → abort with `collision` (unless the motion is a phase that expects contact, e.g. the final approach to a grasp, where finger contact is excluded by the env).
- Named joint configs in `robot.yaml`: `stow` (compact, for driving), `ready` (above-workspace, start/end of skills), `carry` (holding an object while driving).

### 5.2 IK
- One IK module used by both reachability (NAV §6.5) and execution, so "reachable" and "executable" agree.
- Full 6-DoF target (position + orientation) for manipulation; reachability keeps its position-only fast path but `pick`/`place` re-check with full pose.
- Joint limits respected; prefer solutions near the seed; nullspace pull toward `ready`.

### 5.3 Safety / interlocks (robot-enforced, not agent-dependent)
- **Base and arm never move at the same time** in v0.1.
- **Before any base motion:** arm moves to `stow`, or to `carry` if holding something. `go_to`/`align` do this automatically.
- **Before any arm skill:** base must be stopped.
- Contact abort as in §5.1. On abort: stop, open nothing, return `collision` with the current state. The agent decides what to do next.

### 5.4 `pick(target)`
1. If holding something → `already_holding`.
2. `GraspProposer.propose(target)` → none → `no_grasp`.
3. Sort by score; keep grasps whose pregrasp and grasp TCP poses both have IK solutions. None → `unreachable`.
4. Execute: joint move to `ready` → joint move to pregrasp IK solution → open gripper → Cartesian line to grasp pose (contact-expected phase) → close gripper → wait `grasp_settle_s`.
5. **Verify:** `gripper_width` > `min_holding_width` (fingers didn't close fully on nothing). Fail → open, retreat, `grasp_failed`.
6. Cartesian lift by `lift_height` → joint move to `ready`. Re-verify width (object didn't slip). Fail → `grasp_failed`.
7. Set held object (§7.2). Success.

### 5.5 `place(surface)`
1. Not holding → `not_holding`.
2. `PlacementProposer.propose(surface, held)` → filter by IK (preplace + place) → none → `no_placement`.
3. Execute: `ready` → preplace (above) → Cartesian line down to place pose (contact-expected) → open gripper → wait → Cartesian retreat up → `ready`.
4. **Verify:** `gripper_width` near fully open and held object no longer between fingers (sim: env reports; real later: width + wrist camera). Clear held object.

### 5.6 `actuate_fixture(fixture, open)`
1. Reachability check from the current base pose to the fixture's handle/position. Fail → `unreachable` with "align first".
2. Arm to `ready` (so it looks plausible in video), then `FixtureActuator.set_state(...)`.
3. Return success with `scripted=True` in the message/data.

## 6. Sim implementations (`envs/robocasa/`)

```
envs/robocasa/
  grasp_proposer.py       # GroundTruthGraspProposer
  placement_proposer.py   # GroundTruthPlacementProposer
  fixture_actuator.py     # ScriptedFixtureActuator
```

### 6.1 GroundTruthGraspProposer
- Look up the sim body matching the detection (same name table as `GroundTruthDetector`); read its pose and bounding box.
- Generate candidates:
  - **Top-down:** TCP above the bbox center at a height so the fingertips sit at ~60% of object height (or at `grasp_depth` below the top for tall objects), gripper axis aligned to the **narrowest horizontal bbox axis**, both 180°-flipped variants.
  - **Side grasps** for tall objects (height > `side_grasp_min_height`): horizontal approach from the robot's side, at mid-height.
- Reject candidates whose required width exceeds the 2F-85 stroke.
- Score: prefer top-down, prefer approach directions facing the robot.
- Pans: grasp the **handle** (handle sub-body/site if the asset exposes one; else skip pans in v0.1 and flag it).

### 6.2 GroundTruthPlacementProposer
- Surface = a detected fixture top (counter, burner, stove) or an object with a flat top (pan, plate, cutting board).
- Sample candidate (x, y) on the surface's top within reach; reject points whose footprint (held object's bbox, padded) overlaps other objects; for a burner or pan target, use its center.
- TCP pose = held object's current grasp offset applied so the object's bottom sits `place_clearance` above the surface; keep the current gripper yaw if feasible, else nearest feasible yaw.

### 6.3 ScriptedFixtureActuator
- Maps fixture/part IDs (`fridge`, `fridge handle_1`, `microwave`, `cabinet_3`, drawers) to RoboCasa fixture objects and calls their open/close helpers.
- **Verify the helpers exist and work for each fixture type we support**; list unsupported types in the PR.

### 6.4 Scene with objects
- The plain Kitchen task spawns no objects. Either switch to a RoboCasa pick-and-place task for testing or add a small spawner that places a configurable set of objects on counters/in the fridge at reset. **Prefer the spawner**, so the same layouts from NAV tests are reused.
- Objects must be added to the detector's name table and the map's `objects` section by `build_map.py`.
- Start set: an egg (or egg-like small object), a mug, a bowl, a pan, a cutting board, a bottle. Use whatever RoboCasa's object registry provides; list substitutions in the PR.

## 7. Agent changes

### 7.1 Tools

All follow NAV_SPEC conventions: `ToolResult`, `wait=True` default, `id`s must be **visible** in `nearby()`.

No object/fixture typing in v0.1: any visible id can be passed to any tool, and the agent decides what makes sense (pick an egg, open a fridge). Wrong-kind calls fail through the normal path with a clear message: `pick("counter_1")` → `no_grasp` ("no graspable object found for counter_1"); `open("egg_1")` → `unsupported` ("egg_1 is not an openable fixture"). Add guardrails only if traces show the agent getting this wrong.

- **`pick(id, wait=True)`**: id must be visible. Errors: `unknown_id`, `not_visible`, `already_holding`, `no_grasp`, `unreachable` ("align to it first"), `grasp_failed`, `collision`, `stopped`, `timeout`.
- **`place(id, wait=True)`**: `id` is the surface/target (counter_1, burner_2, pan_1). Errors: `unknown_id`, `not_visible`, `not_holding`, `no_placement`, `unreachable`, `collision`, `stopped`, `timeout`.
- **`open(id, wait=True)` / `close(id, wait=True)`**: fixture or handle id. Result includes `"scripted": true`. Errors: `unknown_id`, `not_visible`, `unreachable`, `unsupported`.
- **`holding()`**: returns the held object id/label or nothing.
- **`stop()`**: unchanged name; now stops base and arm.

`pick`/`place`/`open` do **not** auto-align. If unreachable, they fail and the agent calls `align` — one way to do things, visible in the trace.

### 7.2 Held object

- On successful `pick`, the agent records `held = {id, label}` and removes that id from detections.
- `nearby()` shows the held object in its own line: `Holding: egg_1`.
- On successful `place`, `held` is cleared; the object reappears in the next `detect`.
- **The held object is excluded from detection** while held (GroundTruthDetector skips the grasped body), so it never shows up as a new visible id the agent could try to align to or pick.
- On `grasp_failed` mid-carry (slip), `held` is cleared and the message says the object may have dropped.

### 7.3 Registry descriptions

Update the workflow description: **`go_to` → `detect` → `align` → `pick` → `go_to` → `detect` → `align` → `place`**. State that `open`/`close` are currently scripted and require being aligned to the fixture or its handle.

## 8. Config

`robot.yaml` additions: named joint configs (`stow`, `ready`, `carry`), joint velocity/accel limits, Cartesian speed, `cart_step`, `max_joint_jump`, IK tolerances, `grasp_settle_s`, `min_holding_width`, `lift_height`, gripper open/closed widths.

`envs/robocasa.yaml` additions: object spawn set and locations, `grasp_depth`, `side_grasp_min_height`, `place_clearance`, supported fixture types.

## 9. Repo layout additions

```
chefbot/
  interface/
    proposers.py          # GraspProposer, PlacementProposer, Grasp, Placement
    fixtures.py           # FixtureActuator, ActuationResult
  robot/
    arm/ ...              # §5
    skills/ ...
    safety.py
  agent/tools/
    manip.py              # pick, place, open, close, holding
  envs/robocasa/
    grasp_proposer.py
    placement_proposer.py
    fixture_actuator.py
    spawn.py              # object spawner (§6.4)
  scripts/
    demo_manip.py
```

## 10. Boundary rules (additions to NAV_SPEC §12)

- `robot/` imports proposers and actuators only via `interface/`; never `envs/`.
- `agent/` never sees grasp poses, joint configs, or TCP frames — only ids and results.
- Proposers and actuators are env-side and may use sim ground truth; nothing else may.

## 11. Acceptance criteria

**Robot layer** (FakeEnv extended with an arm integrator, gripper width, and a fake graspable object):
- Joint moves reach targets within tolerance and respect limits; Cartesian line moves track a straight line within `cart_tol`.
- IK: reachable pose solved; unreachable pose rejected; solutions within joint limits.
- `arm_in_contact` mid-move → `collision`, arm stops within one tick.
- Interlock: calling `go_to_pose` while holding moves the arm to `carry` first; arm skills refuse while the base is moving.
- `pick` on a fake object → success, `get_held()` set; on "nothing there" → `grasp_failed` and gripper reopened.
- `stop()` during a pick halts the arm.

**Agent** (FakeRobot): error paths for every tool; held-object bookkeeping; `nearby()` shows `Holding:`.

**Integration** (RoboCasa, ≥2 layouts, in-process and over `RemoteRobot`):
- For each object in the start set placed on a counter: `go_to` → `detect` → `align` → `pick` succeeds in ≥ 4/5 trials (report per-object rates).
- `pick` without `align` from far away → `unreachable`.
- Full carry: pick a mug from counter_1 → `go_to("stove")` (arm goes to `carry` automatically) → `detect` → `align` → `place(burner_1)`; mug ends within 5 cm of burner center.
- `open("fridge")` after aligning to the fridge handle → door open, result `scripted: true`; `close` reverses it.
- `scripts/demo_manip.py` runs: `go_to("fridge") → detect(["fridge handle"]) → align → open → detect(["egg"]) → align → pick → close → go_to("counter_1") → detect([...]) → align → place` and saves a video + JSON trace like the nav demo.
- No unexpected arm collisions with fixtures during any successful run.

## 12. Open questions for the implementer to answer first

1. Does RoboCasa's mobile robot accept `Robotiq85Gripper`? Any changes to its grip site naming?
2. Does `arm_in_contact` (excluding finger–target contacts) come cheaply from MuJoCo contacts? Define precisely which geom pairs count.
3. Which RoboCasa objects expose handle sub-bodies/sites (pans, mugs)?
4. Which fixture types have working `open_door`/`close_door`/drawer helpers in our RoboCasa version?
5. Panda (sim) vs FR3 (real) kinematics: does the sim robot use Panda joint limits? If so, use the sim robot's limits in sim config and FR3's for real, and note differences.

## 13. Future work

- **Real door/drawer opening:** grasp the handle, then follow the hinge/slide trajectory under impedance control (hinge axis from sim now; from the map or estimation on the real robot). This is the first candidate for a VLA tool.
- **Obstacle-aware arm planning** (cuRobo, or MoveIt-style sampling planners) instead of ready → pregrasp → line moves.
- **Learned proposers:** SAM/GroundingDINO segmentation + GraspGen/AnyGrasp grasps; VLM/Molmo pointing for placement. Isaac or real robot only (MuJoCo renders break them).
- **π0.5 as a tool** alongside the primitives, VoLo-style, for contact-rich steps (cracking, stirring, flipping).
- Placing inside containers/shelves, pouring, tool grasps (spatula, tongs), bimanual-free regrasps.
- Grasp/place verification from the wrist camera rather than gripper width alone.
- Object/fixture typing in `nearby()` (`not_pickable` / `not_openable` errors) if the agent misuses tools.
- Force-limited gripper control for eggs (continuous width/force instead of binary).
