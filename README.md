# ChefBot

Navigation tool layer for the ChefBot cooking agent: the tools an LLM calls to
move a mobile manipulator around a kitchen. See [NAV_SPEC.md](NAV_SPEC.md) for
the design.

```
agent  (chefbot/agent)   tools, semantic map, detections, nearby view    env-agnostic
  |  RobotAPI + Detector  (in-process, or RemoteRobot over a websocket)
robot  (chefbot/robot)   occupancy grid, A*, standoff poses, base control, FR3 reachability
  |  Env  (always in-process)
envs   (chefbot/envs)    robocasa today; isaac and real are placeholders
```

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
pytest chefbot/tests                     # adds the RoboCasa integration tests: about 6 minutes
```

## Notes on the RoboCasa backend

- The robot is RoboCasa's `PandaOmron`. Its base is holonomic with velocity
  actuators. The base joints have heavy friction (a command of v yields about
  v - 0.25 m/s), which `RoboCasaEnv` cancels with a feedforward term so
  `set_base_velocity` delivers what was asked for.
- The sim arm is a Panda; reachability uses the FR3 model
  (`chefbot/robot/kinematics/assets/`) placed at the sim robot's arm mount.
- The shoulder camera is RoboCasa's `robot0_frontview` (0.5 m behind the base
  center, 1.65 m up). The ground-truth detector checks the camera frustum and
  occlusion by the scene, but looks past the robot's own arm.
- Simulated time only advances while a motion is running.
