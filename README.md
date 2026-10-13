# ChefBot

Tool layer for the ChefBot cooking agent: the tools an LLM calls to move a
mobile manipulator around a kitchen and to pick things up, put them down and
open fixtures. See [NAV_SPEC.md](NAV_SPEC.md) and [MANIP_SPEC.md](MANIP_SPEC.md)
for the design.

```
agent  (chefbot/agent)   tools, semantic map, detections, nearby view, held object     env-agnostic
  |  RobotAPI + Detector  (in-process, or RemoteRobot over a websocket)
robot  (chefbot/robot)   grid, A*, standoff poses, base control, arm IK and control, pick / place / open
  |  Env + GraspProposer, PlacementProposer, FixtureActuator  (always in-process)
envs   (chefbot/envs)    robocasa today; isaac and real are placeholders
```

Tools: `go_to`, `detect`, `nearby`, `align`, `pick`, `place`, `open`, `close`,
`holding`, `wait`, `get_status`, `stop`, `get_pose`. The workflow they are built
around is `go_to -> detect -> align -> pick -> go_to -> detect -> align -> place`.

## Setup

Python 3.10+. The agent and robot layers need no simulator:

```sh
pip install -e ".[robot,dev]"       # agent + robot layers and their tests
```

The RoboCasa backend also needs robosuite and robocasa from source, on a native
arm64 Python on Apple Silicon:

```sh
CONDA_SUBDIR=osx-arm64 conda create -n chefbot -c conda-forge python=3.11
conda activate chefbot
git clone https://github.com/ARISE-Initiative/robosuite && pip install -e robosuite
git clone https://github.com/robocasa/robocasa && pip install -e robocasa
python robocasa/robocasa/scripts/download_kitchen_assets.py      # about 10 GB
pip install "mujoco==3.3.1" mink                                  # robocasa pins mujoco 3.3.1
pip install -e ".[robot,dev]"
```

## Run

```sh
# Scripted trace, everything in one process
python -m chefbot.scripts.demo_nav --env robocasa

# Same trace with the agent talking to the robot layer over a websocket
python -m chefbot.envs.robocasa.serve                              # terminal 1
python -m chefbot.scripts.demo_nav --remote ws://127.0.0.1:8765    # terminal 2

# Manipulation trace: open the fridge, take the egg out, close it, put the egg
# on a counter. Saves rollouts/manip_layoutNNN_styleNNN.mp4 and .json
python -m chefbot.scripts.demo_manip --env robocasa
python -m chefbot.scripts.demo_manip --remote ws://127.0.0.1:8765  # against a running server (no video)

# Record a rollout to rollouts/<trace>_layoutNNN_styleNNN.mp4 (+ .json tool log);
# traces: demo, tour, stop, manip
python -m chefbot.scripts.record_rollout --trace tour --layout 2 --style 2

# Rebuild a kitchen's map files (maps/robocasa_layoutNNN_styleNNN/)
python -m chefbot.envs.robocasa.build_map --layout 2 --style 2
```

Using the tools from an agent:

```python
from chefbot.agent.config import AgentConfig
from chefbot.agent.tools.registry import ToolRegistry
from chefbot.agent.world.map import WorldMap

registry = ToolRegistry(robot, detector, WorldMap.load("semantic_map.yaml"), AgentConfig.load())
registry.schemas                                   # OpenAI-style function schemas for the LLM
registry.dispatch("go_to", {"place": "stove"})     # -> ToolResult(success, error, message, data)
```

`robot` and `detector` are any `RobotAPI` and `Detector`: the in-process pair
from `chefbot.envs.robocasa.serve.build_stack()`, or `RemoteRobot(url)` and
`RemoteDetector(url)`.

## Tests

```sh
pytest chefbot/tests -m "not robocasa"   # agent, robot, transport, boundaries: seconds, no simulator
pytest chefbot/tests                     # adds the RoboCasa integration tests: about 25 minutes
```

The slowest RoboCasa test picks each object five times per layout and prints the
success rates (`-s` shows them); deselect it with `-k "not rates"`.

## Notes on the RoboCasa backend

- The robot is RoboCasa's `PandaOmron`. Its base is holonomic with velocity
  actuators. The base joints have heavy friction (a command of v yields about
  v - 0.25 m/s), which `RoboCasaEnv` cancels with a feedforward term so
  `set_base_velocity` delivers what was asked for.
- The sim arm is a Panda. IK and reachability use the FR3 model
  (`chefbot/robot/kinematics/assets/`) placed at the sim robot's arm mount; the
  two arms have identical kinematics (checked to 1e-15 m on random poses) but
  different joint limits, so `robot.yaml` carries the Panda's limits for sim.
- The shoulder camera is RoboCasa's `robot0_frontview` (0.5 m behind the base
  center, 1.65 m up). The ground-truth detector checks the camera frustum and
  occlusion by the scene, but looks past the robot's own arm.
- Simulated time only advances while a motion is running.

## Notes on manipulation (MANIP_SPEC v0.1)

What is real and what is faked: arm motion, IK, the gripper and contact are
simulated physics driven by env-agnostic code in `chefbot/robot`. The faked parts
are perception (where an object is, how to grasp it, where to put it: the
ground-truth proposers in `chefbot/envs/robocasa`) and opening fixtures, which
sets the door joint directly (`scripted: true` in the tool result).

Answers to the spec's open questions (§12):

1. **Gripper.** RoboCasa's mobile robot accepts robosuite's `Robotiq85Gripper`,
   but that model is not usable: its fingers jam on their own linkage at a gap of
   about 6 cm (an egg or a bottle cannot be gripped) and the linkage goes unstable
   on repeated closes. The sim uses MuJoCo Menagerie's 2F-85 instead, ported to a
   robosuite gripper (`envs/robocasa/gripper.py`, `assets/robotiq_2f85/`): real
   4-bar linkage, 85 mm stroke, closes fully. Its grip site is 0.155 m past the
   flange, turned -45 degrees about the approach axis (`tcp` in `robot.yaml`).
2. **`arm_in_contact`.** Cheap, from the MuJoCo contact list: arm links 2-7, the
   hand and the gripper body touching the world or the mobile base. Finger geoms
   never count, and neither does contact with an object touching both pads (the
   held object). `base_in_contact` is now the mobile base only.
3. **Handles.** RoboCasa objects are single bodies with no handle sites. The
   grasp proposer finds a pan's handle from its geometry (a thin run at one end
   of the long axis) and grasps bowls by the rim.
4. **Fixtures.** Scripted open / close works for the fridge (and its freezer),
   microwave, dishwasher, oven door, hinge cabinets and drawers; a test opens and
   closes one of each kind of handle the detector knows.
5. **Panda vs FR3.** Same kinematics, different limits (above). The sim's start
   pose is outside the FR3's limits on joints 4 and 6, so sim config uses the
   Panda's; delete `joint_limits` from `robot.yaml` for the real arm.

Things that differ from the spec, or that it did not anticipate:

- **`open` and `close` can move the base.** A door swings through where the
  robot stands to reach its handle, so the robot first steps out of the door's
  arc, and an open door is added to the planning grid until it is closed.
- **`align` is stricter.** A standoff pose must let the gripper come down on the
  target from 20 cm above (or reach in from 20 cm in front, for targets above
  1.2 m). Position-only reach left the arm too stretched to grasp.
- **Shelves and wall cabinets.** An object with less than 35 cm of headroom is
  grasped from the front, gripper angled down, then backed straight out. The
  Panda cannot hold its gripper level at counter height (wrist limit), which is
  why the egg sits on the fridge's upper shelf and why front and side grasps are
  angled. A from-above grasp is shortened, or dropped for an angled one, when a
  wall cabinet leaves the arm no room over the object (a tall bottle).
- **Big surfaces.** A detection of a counter carries the free, flat spot nearest
  the robot (`near_point_world`); `align` goes there instead of to the counter's
  middle, which may be a sink, and `place` sets the object down as close to that
  spot as it can.
- **Objects on the map.** `go_to("mug")` stands in front of the mug, not the
  middle of its counter. Picking an object removes it from the map; put down
  somewhere else it comes back from `detect` under a fresh id (`mug_1`).
- **Start set.** Egg, mug, bowl (rim), bottle and pan (handle) can be picked;
  the cutting board is a place target only (`pick` returns `no_grasp`).
- **Physics settings.** MuJoCo's no-slip solver is on (`noslip_iterations`):
  without it a held mug creeps out of the fingers within seconds. With it, the
  base's joint friction is a hard threshold, so the mobile base's own geoms are
  frictionless and the friction feedforward applies in full at any speed.
- **Robot layer layout.** `actuate_fixture` lives in `robot/robot.py` rather than
  `robot/skills/actuate.py`: stepping out of a door's way needs the planner.
- **No obstacle-aware arm planning** (spec §13): the arm moves ready -> approach
  -> straight line. In tight spots (the small counter beside the fridge) it can
  touch a fixture, which stops the skill with `collision` and the object still
  in hand.
- `stop_base` is now `stop`, and `tcp_offset` is replaced by `tcp`, as MANIP_SPEC
  says; NAV_SPEC.md still shows the old names.

