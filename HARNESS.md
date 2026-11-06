# The agent harness: what changed from the agent loop on main

`main` has the kitchen agent loop of AGENT_SPEC v0.1 (Olivia's): one LLM works through a plan one tool call per
turn, a judge LLM checks each step's `done_when`, a rolling window of the last 8 calls plus a TASK STATE rebuilt every
turn is its context, and it runs in the symbolic kitchen and in RoboCasa. This file says what each branch built on it
and why. Branch `agents-skills-registry` (this one) is below; `harness-memory-planner` adds its own section on top.

## The loop on main (unchanged in its behavior)

| Part | On main | Where |
|---|---|---|
| Loop | one tool call per turn; the window of the last 8 calls; TASK STATE (plan, now, perception, kitchen, events) last | `agent/loop/runner.py`, `context.py` |
| Executor prompt | how to act, find things, perception freshness, motion and waiting, the plan, safety, ending | `agent/loop/prompts/system.md` (+ `handheld.md`, `trigger_tools.md`) |
| Judge | Haiku checks `done_when` against the simulator's state, replies `{"passed", "reason"}` | `agent/check/checker.py` |
| Tools | 30 tool schemas and their handlers, written in one file | `agent/tools/registry.py` |
| Settings | the tools' settings and the loop's | `config/agent.yaml` |
| Runs | `run_agent`, `eval_agent` on `symbolic` and `robocasa` | `scripts/` |

## Branch agents-skills-registry: one loop, many agents, skills from one registry, sim and real

Each step kept every agent run byte-identical (the first LLM request, the 30 tool schemas for three robots, the
scripted symbolic run) unless the step's point was to change it; commits are in `git log main..agents-skills-registry`.

### a) Evolving the loop so it runs in sim and on the real robot

- **The loop became a named harness, `plan_loop`** (`agent/harnesses/plan_loop/`: `runner.py`, `context.py`,
  `perception.py`, moved from `agent/loop/`). The loop is the same code; an agent names the harness it runs in.
- **A run is a session: one agent, one environment, one robot API** (`agents/session.py`). The session offers the
  skills the agent selects that can run in that environment and speak that API, and the tools the agent lists that work
  with it. A mismatch (an API the harness doesn't run, two skills with one name) is refused before anything is built;
  `start.json` records the session.
- **Two robot APIs.** `robot-api-v1`: a skill is a function in the agent's process (the 14 sim skills).
  `chefbot-v1`: a skill is a sandboxed program run as one episode on a host (the lab's skills and the real robot's).
  `plan_loop` runs both: on the real robot each skill call is one sandboxed episode on the real host, kept open for the
  run and continuing where the last skill left it (`runtime/agent_skills.py`); its result goes to the LLM as is.
- **The real robot** (`run_agent --env real [--dry-run] [--confirm]`, `environments/real/agent_world.py`): everything
  runs on the laptop against the onboard server, which is not changed. A robot that moves only through its skills, no
  detector, no map, the wall clock, no ground truth: a completed step is taken as done and the log says so
  (`UncheckedJudge`). Before the robot is touched, each skill's requirements are checked (a missing `GEMINI_API_KEY`
  stops the run with that message).
- **Safety for the real arm.** The arm controller refuses a planned move whose path goes into the robot's own body
  (the pole, the camera clamp) before anything is sent, by the same rule the real host's guard uses
  (`robot/control/constraints.py`); the real speed limits are enforced (`tidybot_fr3` `control.real_limits`).
  `ready/v1` moves the arm to the ready pose around the robot's body: on the robot, `pick_up/gemini_v1`'s straight move
  home grazed the pole camera's keep-out and the guard refused it.
- **A second agent, `real_arm`**: the arm only, `ready/v1` and `pick_up/gemini_v1` (Gemini Robotics ER on the wrist
  camera), the plan's tools, its own 23-line prompt, real time, 20 turns, $1. Status: the read-only check passed, the
  dry run found the pole D455 missing on USB (so: wrist camera only) and the keep-out graze (so: `ready/v1`); a live
  pick is not run yet.

### b) Designing the agent's prompts, tool descriptions and registries

- **An agent is a definition, not code** (`agents/<name>/agent.yaml`): its harness, the robot APIs it runs with, its
  prompts folder, the skills it gets (patterns: `*/robotapi_v1`), the tools it gets (by name), the tools' settings and
  the loop's (model, judge model, window, limits). `config/agent.yaml` is gone. `run_agent --agent NAME`.
- **Prompts live with the agent** (`agents/kitchen_planner/prompts/`: `system.md`, `handheld.md`, `trigger_tools.md`;
  `agents/real_arm/prompts/system.md`). `system_prompt()` adds a section for each kind of tool offered.
- **Tools that don't move the robot are a registry, one YAML file each** (`tools/<group>/<name>.yaml`: perception,
  motion, plan, time): the description the LLM reads, the arguments with their descriptions, the handler
  (`entry: owner.method`), the robot capabilities it needs, the robot APIs it works with.
- **Skills that move the robot are in the same registry as the lab's skills** (`skills/<name>/<version>/skill.yaml`):
  `interface` (robot API), `envs` (where it can run: symbolic, robocasa, mujoco_kitchen, real), `sandboxed`, `needs`
  (capabilities), `description` and `args` (what the LLM reads), `entry` (its code). A manifest that contradicts itself
  is refused when loaded. The 14 sim skills are `*/robotapi_v1`; the agent's pick is `pick_up/robotapi_v1`, beside the
  lab's `pick_up` versions.
- **What reaches the LLM**: of each tool or skill, only `description` and the arguments' descriptions, as a function
  schema. Whether it is offered at all: the agent selects it, the session's environment and API can run it, the robot
  has what it `needs`, the job doesn't exclude it.

### Unchanged from main

The executor's system prompt (moved, and `pick` renamed `pick_up` in it), the judge and its prompt, the TASK STATE,
the window of 8, the tools' code, the symbolic and RoboCasa environments.
