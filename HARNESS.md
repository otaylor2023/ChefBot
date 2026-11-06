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

## Branch harness-memory-planner: from a user's request to a plan, memory, evaluation, and seeing it work

Built on `agents-skills-registry`. Every change was measured before and after on evaluation sets modeled on the
robot-memory papers (MessyMem, BUMBLE, the Orchestration Gap, SayPlan, HW1's personas): `EVALUATION.md`.

### a) The loop, adapted further

- **A planner in front of the loop** (`agent/plan/planner.py`, `run_agent --plan-from prompt [--prompt "..."]`).
  On main a run could only start from a plan written in the job file. Now a second LLM turns the user's sentence into
  the plan, with three tools: `ask_user` (one question when the dish or where it ends up isn't said), `cannot_do`
  (the kitchen lacks something the dish needs), `submit_plan`. Code checks the plan (ids, dependencies, cycles, every
  `done_when` a fact about the kitchen, not a robot action) and sends problems back. `request_replan` calls it again
  with the executor's reason, keeping the steps already done. In evaluations a simulated user (Haiku, the job's
  `user:` field) answers its questions.
- **What the executor remembers beyond its 8-call window** (`plan_loop/context.py`, `tools/plan_tools.py`):
  each step keeps its failed attempts (`tried:`) and how many times the check refused it; `complete` refused twice is
  blocked until a call changes the kitchen; LEARNED BY DOING keeps, for the whole run, what failed on what and what
  made it work; SEARCHING lists only what the agent asked for by name; `finish` is refused once while a door the robot
  opened is still open.
- **The kitchen remembered across runs** (`run_agent --memory PATH`): the map as the last run left it, loaded instead
  of the environment's and saved at the end.
- **The window stays 8**: 4 loses track (6/12), 8, 16 and 32 differ by at most one run (10, 11, 11 of 12) and 32 costs
  80% more; in RoboCasa a failing call was repeated after leaving the window once in 282 calls.

### b) Prompts and tool descriptions

| LLM | System prompt | Its user message | Its tools | Written |
|---|---|---|---|---|
| Planner | `agents/kitchen_planner/prompts/planner.md` (+ `planner_narrate.md`) | REQUEST, KITCHEN (what the robot knows), ROBOT (one line per skill) | ask_user, cannot_do, submit_plan | this branch |
| Executor | `system.md` + `handheld.md` / `trigger_tools.md` (+ `narrate.md`) | the last 8 calls, then the TASK STATE | the session's skills and tools (28 here) | main; unchanged here |
| Judge | `JUDGE_PROMPT` in `agent/check/checker.py` | CONDITION, STATE | none: JSON | main; unchanged |
| Simulated user | `PROMPT` in `agent/plan/simulated_user.py` | the planner's question | none | this branch |

- `planner.md` tells it how cooking works in this kitchen (a whole egg fries; scrambling needs the cracker, a bowl and
  stirring; typical times and when food overcooks), what a good plan is (steps that change the kitchen, `done_when` a
  checkable fact, the burner off, doors closed, no staging), when to ask and when to refuse. Each rule came from a
  failure in the requests evaluation: 29/42, then 32/42, then 39/42, then 41/42.
- The planner's ROBOT list is the first sentence of each skill's own description (`skills_text`): the descriptions
  in the registry are the one source for both LLMs.
- **Narration** (`run_agent --narrate`, on in the chat): `narrate.md` and `planner_narrate.md` ask both LLMs to say
  what they do and why before each call. Off in evaluations, so their prompts and results are unchanged.

### c) Seeing it work

| What | How |
|---|---|
| Talk to the agent live, as in HW1's chat | `python -m chefbot.scripts.chat [--env robocasa --video]` |
| A run as who said what to whom | `runs/<run>/conversation.html` (every run; `view_conversation`) |
| The request, questions, answers and plan in the report | `runs/<run>/report.html`, "User request" |
| A video with the conversation beside it | `python -m chefbot.scripts.chat_video runs/<run>` → `chat_video.mp4` |
| Everything the LLMs saw and returned | `runs/<run>/requests/*.json`, `planner.json` (raw outputs, the whole exchange), `checks/` |

### Results (symbolic kitchen; RoboCasa for the videos)

Requests planned from the user's sentence 41/42 (hand-written plans 33/34); ambiguous requests 6/6 with a user to ask,
0/6 without; the impossible request refused 2/2; recovery with injected failures 21/21; finding moved objects 12/12
(8/12 before); with the kitchen remembered, the second task in a sequence takes 17 turns instead of 31. RoboCasa:
fried sausage from the typed request, success, 50 turns, $0.34 (`runs/2026-11-06_102238_fried_sausage_robocasa_s0`).
