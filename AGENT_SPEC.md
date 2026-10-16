# ChefBot Agent Loop — Implementation Spec (v0.1, sim)

Builds on NAV_SPEC.md and MANIP_SPEC.md and the current repo (see "ChefBot Project Spec"). Same layers and boundary rules: the agent decides, the robot plans and controls, the env actuates. The agent layer still never sees poses, grasps, joint states or simulator ground truth.

## 1. Goal

An LLM agent that executes a cooking plan through the existing 13 tools, tracks progress on a plan it can edit, verifies steps before marking them done, handles things that take time (timers, monitors, events), can pause one step to deal with another, and leaves a log that is easy to review afterward.

**In v0.1:** plan store, step verification, timers/monitors/events, event-aware waiting, the LLM loop, context assembly, run logging + a run viewer, a time-based cook model and burner on/off in sim, hand-written test plans.

**Not in v0.1:** the planning agent (recipe search, grounding, plan generation), exploration/map building, images in the agent's context, new manipulation skills (stir, pour, crack). See §15.

## 2. Where things live

```
chefbot/
  interface/
    observer.py          # Observer ABC: what a checker can look at (§5.1)
    clock.py             # Clock ABC: now(), used for timers (§3)
  agent/
    plan/
      model.py           # Plan, Step, statuses, transitions (§4)
      store.py           # reference plan (immutable) + live plan + edit log
    check/
      checker.py         # StateChecker: judge(condition, observation) (§5)
    events/
      bus.py             # event queue; timers, monitors, safety monitors (§6)
    loop/
      llm.py             # OpenRouter client (OpenAI-compatible tool calling)
      context.py         # builds each turn's messages (§9)
      perception.py      # perception block + freshness (§9.3)
      runner.py          # the agent loop (§8)
      prompts/system.md  # system prompt (§9.1)
    tools/
      plan_tools.py      # complete, set_status, add_step, skip_step, request_replan, finish (§7.2)
      time_tools.py      # set_timer, cancel_timer, watch (§7.3)
      registry.py        # existing; registers the new tools too
    logging/
      run_log.py         # structured run log writer (§11)
  robot/
    clock.py             # sim/real clock; idle stepping (§3)
  envs/robocasa/
    observer.py          # GroundTruthObserver: state as text (§5.2)
    cook_model.py        # time-based cooking state (§5.3)
    fixture_actuator.py  # existing; add burner on/off (§5.4)
  scripts/
    run_agent.py         # run a task (§12)
    view_run.py          # render a run log (§11.4)
  tasks/                 # hand-written test plans (§13)
```

Boundary additions (to `tests/test_boundaries.py`):
- `Observer` output goes **only** to the `StateChecker` and the run log. It never enters the agent's LLM context. Test: the context builder imports nothing from `check/` or `interface/observer.py`.
- The LLM client lives only under `agent/loop/` and `agent/check/`.

## 3. Time

Timers and cooking need a clock that keeps running when the robot isn't moving. Today sim time only advances inside motions.

- **`Clock.now()`** returns seconds since run start. Sim: sim time. Real: wall time.
- **Idle stepping:** the robot layer steps the env while no motion is running, so burners, cooking and timers progress.
- **Two time modes** (`agent.yaml: time_mode`):
  - **`paused` (default for development):** sim time is frozen while the LLM is thinking. It advances only during motions and during `wait()`, which fast-forwards the sim until the next event or timeout. Deterministic and fast: a 2-minute cook takes seconds of wall time.
  - **`realtime`:** the sim runs continuously at wall-clock rate, including during LLM calls. The egg keeps cooking while the agent thinks. Use for realism tests once the loop works.
- All timers, monitors, cook state and log timestamps use `Clock.now()`. The agent context shows no clock; only durations on steps (§9.2).

## 4. Plan

### 4.1 Model

```yaml
goal: "Fried egg, served on a plate"
steps:
  - id: s3
    goal: "Cook the egg in the pan until the whites are set"
    done_when: "the egg is in the pan, the pan is on a lit burner, and the whites are set"
    depends_on: [s1, s2]
    hint: "set a ~2.5 min timer and watch this step"     # optional, from the planner
    status: pending
    notes: []                 # agent notes, resume notes, check failures
    log: null                 # one-line summary written when the step ends or pauses
    added_because: null       # set for steps added mid-run
```

`done_when` is plain text. No predicates.

### 4.2 Statuses and transitions

| Status | Meaning |
|---|---|
| pending | not started |
| active | the robot is working on it now |
| pausing | the agent has decided to pause it and is winding it down (e.g. putting the mixer down) |
| paused | started, stopped partway; has a resume note |
| waiting | progressing without the robot (cooking, a timer running) |
| done | passed its check |
| failed | the agent gave up on it (with reason) |
| skipped | not needed (with reason) |

Rules (enforced by the store, errors returned to the agent):
- **At most one step is `active` or `pausing`.** Setting a second one active → `another_step_active` (message names it).
- `pending → active` requires all `depends_on` steps `done` or `skipped` → else `dependencies_not_met`.
- `active → pausing → paused` (paused requires a resume note). `active → paused` directly is allowed if nothing needs winding down.
- `paused → active`, `waiting → active`, `active → waiting` are allowed.
- **`done` only via `complete()`** (§7.2). `set_status(..., "done")` → `use_complete`.
- `failed` and `skipped` require a reason.

### 4.3 Reference plan and live plan

- The **reference plan** is loaded at run start and never changes.
- The **live plan** starts as a copy. Every change (status, note, add, skip) is appended to an **edit log** with time, step, change and reason.
- Both, plus the edit log, are written to the run log, so a run can be compared to its reference (steps added, skipped, reordered, retried).

## 5. Verification

### 5.1 Interfaces

```python
class Observer(ABC):                       # env-provided, like Detector
    def observe(self) -> Observation: ...
# Observation: text: str | None, images: list[RGB] | None, extra: dict

class StateChecker:                        # agent-side
    def check(self, condition: str, obs: Observation) -> CheckResult: ...
# CheckResult: passed: bool, reason: str, confidence: float | None
```

- v0.1 checker: an LLM judge (`agent.yaml: judge_model`, a small fast model) given the condition and `obs.text`. It returns `{passed, reason}` as JSON.
- Later: a VLM judge over `obs.images`, plus proprioception/force in `obs.extra`. Same interface.

### 5.2 GroundTruthObserver (sim)

Renders the current ground truth as short text: relations, not coordinates.

```
held: nothing
egg: inside pan
pan: on burner_front_left
burner_front_left: on (for 2:10)
burner_front_right: off
egg: cook state = set (2:10 of heat)
plate: in cabinet_2
cabinet_2: open
fridge: closed
mug: on counter_1
```

Relations to compute (from contacts and bounding boxes): `held`, `inside <container>`, `on <surface or burner>`, fixture `open/closed`, burner `on/off` with duration, cook state for food. Names are stable sim names mapped to ChefBot labels (not detection ids, which change). If a relation the plans need is missing, the judge can't see it: add it here.

### 5.3 Cook model (sim)

- A food item **heats** while it is inside a pan (or pot) that is on a burner that is on. Heat time accumulates; it never decreases.
- Cook state from heat time, thresholds per food type in `robocasa.yaml`:
  `raw → cooking → set → overcooked` (egg defaults: cooking > 10 s, set > 120 s, overcooked > 300 s).
- No physics, no visual change. The state only appears in the observer's text.

### 5.4 Burners (sim)

- Extend `FixtureActuator` with `set_power(burner, on: bool)` and state in `get_state`.
- **Verify RoboCasa's stove exposes burner/knob state** we can set and read; if not, keep burner state in our env (a dict), toggled by the actuator. Either way the agent uses `turn_on(id)` / `turn_off(id)` on a visible burner id, aligned like `open`. Results say `scripted: true`.

## 6. Timers, monitors, events

### 6.1 Events

```python
Event: id, time, kind ("timer" | "monitor" | "safety" | "motion_done"), step_id?, message, data
```

Events go on one queue owned by the agent layer. They are delivered:
- **into `wait()`**: `wait` returns on the first event (§7.4);
- **into a blocking motion tool**: it returns early with `status: "moving"` and the event attached; the motion keeps going unless the agent calls `stop()`;
- **otherwise** at the start of the next turn, in the task state's `EVENTS SINCE LAST TURN`.

Events never preempt motion by themselves, except safety events (§6.4).

### 6.2 Timers
`set_timer(name, seconds, step_id?)` → fires once at `now + seconds`. `cancel_timer(name)`. Active timers show on their step in the task state ("timer 'egg' 0:45 left").

### 6.3 Monitors
`watch(step_id, every_s, timeout_s)` → every `every_s`, run `StateChecker.check(step.done_when, Observer.observe())`.
- Pass → event `monitor passed`, monitor stops. **The step is not auto-completed**: the agent still calls `complete()`. (It will pass immediately.)
- Timeout → event `monitor timeout` with the last reason.
- Monitor checks are logged like any other check (§11).

### 6.4 Safety monitors (always on, not agent-controlled)
- Burner on with nothing on it for > `safety.empty_burner_s` (default 30 s) → event, and the robot layer turns it off.
- Food `overcooked` → event.
- Run-level limits (§8.3) → event, then stop.

## 7. Tools

Existing 13 tools unchanged, except `wait` (§7.4). New tools below. All return `ToolResult`.

### 7.1 Kitchen
- `turn_on(id)` / `turn_off(id)`: burners; visible id; scripted. Errors: `unknown_id`, `not_visible`, `unreachable`, `unsupported`.

### 7.2 Plan
- `complete(step_id)`: runs the check on `done_when`. Pass → `done`, the agent's last note becomes the step's log line (or "completed" if none), result includes the judge's reason. Fail → status unchanged, error `check_failed` with the judge's reason ("egg is held, not in the pan").
- `set_status(step_id, status, note?)`: transitions per §4.2. `paused` requires a note (the resume note). `failed` requires a note.
- `add_step(after, goal, done_when, reason, depends_on?)`: inserts a step; `added_because = reason`.
- `skip_step(step_id, reason)`.
- `request_replan(reason)`: v0.1 has no planning agent. Logs the request, pauses the run, and asks the operator in the terminal (continue / abort / edit plan file and reload). Later this calls the planning agent.
- `finish(summary)`: ends the run. Allowed only if every step is `done`, `skipped` or `failed`; else `steps_remaining` listing them. `finish(summary, abandon=true)` ends anyway (logged as abandoned).

### 7.3 Time
- `set_timer`, `cancel_timer`, `watch` (§6).

### 7.4 `wait(timeout_s?)` (changed)
Returns on the first of: current motion ends, any event, timeout. With no motion and no timers/monitors running and no timeout → returns immediately with `nothing_to_wait_for` (prevents a dead loop).

## 8. The loop

### 8.1 Turn

```
loop:
  messages = context.build()                   # §9
  resp = llm.chat(messages, tools=registry.schemas, tool_choice=auto, parallel_tool_calls=false)
  if resp has no tool call:
      append resp text; append "Respond with exactly one tool call."; retries += 1 (max 2, then error)
      continue
  call = resp.tool_calls[0]
  validate args against schema → on failure, tool result = invalid_args (no dispatch)
  result = registry.dispatch(call.name, call.args)
  log turn (§11); update perception block; update plan/event state
  if call.name == "finish" and result.success: break
```

- **One tool call per turn** (`parallel_tool_calls=false`; if a provider returns several anyway, run the first and return `one_call_per_turn` for the rest).
- **Text is optional.** Any text before the call is logged and kept in the rolling window as the agent's reasoning.
- **`tool_choice=auto`** so models with extended thinking work; the re-prompt above enforces a tool call.
- Bad tool names, invalid args and malformed JSON become error results, never crashes.

### 8.2 Model
- `agent.yaml: model` via OpenRouter; pin provider + model for reproducibility. Start with a strong tool-use model.
- `judge_model`: a small fast model.
- Both temperatures configurable (default 0 for the judge).

### 8.3 Limits
`max_turns`, `max_llm_cost_usd`, `max_sim_time_s`, `max_wall_time_s`, `max_consecutive_errors` (same error N times in a row). Hitting one → safety event, robot `stop()`, burners off, run ends with `limit_reached`.

## 9. Context (rebuilt every turn)

Order matters for prompt caching: [1] and [2] never change within a run; [3] is a rolling window; [4] is always the last message and only the newest copy is kept.

### 9.1 [1] System prompt (static)
`prompts/system.md`. Content as in CONTEXT_EXAMPLE.md: how to act (go_to → detect → align → pick/place/open), only ids visible in the latest perception can be acted on, perception freshness, motion and waiting (blocking calls wake on events; use `wait` instead of polling), plan statuses and rules (`complete` runs a check; pausing → paused with a resume note; one active step; use timers/monitors for things that take time; `add_step`/`skip_step` with reasons; `request_replan` instead of changing the dish), safety (never leave a burner on with nothing on it), `finish` to end.

### 9.2 [2] Tool schemas (static)
From the registry.

### 9.3 [3] Rolling window
- The last `N` (default 8) assistant tool calls + tool results, with any reasoning text.
- Older ones are dropped; step log lines and the perception block carry what matters.

### 9.4 [4] Task state (last message)

```
TASK STATE
GOAL: ...
PLAN
  s1 done     <goal>        log: <one line>
  s3 waiting  <goal>        done_when: ...   cooking 2:10 · timer 'egg' 0:20 left · monitor every 20s
  s4 paused   <goal>        resume: <note>
  s5 active   <goal>        done_when: ...
  s6 pending  <goal>        depends: s3, s5
NOW
  holding: plate_1 · motion: go_to("counter_1") in progress
PERCEPTION  (latest: detect(["plate"]) at cabinet_2 · possibly stale: moved since)
  visible:      ...
  not visible:  ...
MAP PLACES: ...
EVENTS SINCE LAST TURN: ...
```

- `done_when` shown only for active, waiting and the next pending step (keeps it short).
- Durations only: time cooking/waiting, timer time left. No clock.

### 9.5 Perception freshness

The latest `detect` / `nearby` result is kept in the PERCEPTION block (not in the rolling window).

| Situation | Block |
|---|---|
| just detected, no movement | fresh |
| > `stale_after_calls` (5) tool calls since | possibly stale (N calls ago) |
| base moved slightly (> 5 cm or > 5°, below the clear thresholds) | possibly stale (moved since) |
| base moved > 0.5 m or > 30° from where it was detected | cleared ("call detect or nearby") |
| new detect / nearby | replaced |

`pick` / `place` update the block the same way the world state does (picked object → held; placed object disappears until the next detect). All thresholds in `agent.yaml`.

## 10. Config (`agent.yaml` additions)

`model`, `judge_model`, provider pins, temperatures, `time_mode`, `window_n`, `stale_after_calls`, `stale_move_m`, `stale_turn_deg`, `clear_move_m`, `clear_turn_deg`, `safety.empty_burner_s`, limits (§8.3), `log_full_requests` (bool).

`robocasa.yaml` additions: cook thresholds per food type, burner state source.

## 11. Run logging and the run viewer

Goal: after a run, see quickly **what the world looked like at the start, what the robot did, and how the world and plan changed after each action**, without wading through prompts.

### 11.1 Run directory

```
runs/2026-10-15_1342_fried_egg/
  run.jsonl          # the timeline: one record per line, small (§11.2)
  start.json         # initial ground-truth state, reference plan, config, model ids, git commit
  end.json           # final ground-truth state, live plan, edit log, outcome, totals
  requests/          # full LLM requests + responses, one file per turn (if log_full_requests)
  checks/            # every judge call: condition, observation text, verdict, reason
  video.mp4          # optional, from the existing rollout recorder
  report.html        # generated by view_run (§11.4)
```

Prompts and judge inputs are kept **out of** `run.jsonl` so the timeline stays readable; records point to them by id.

### 11.2 `run.jsonl` records

Every record: `t` (Clock.now), `turn`, `kind`. Kinds:

- **`action`**, one per tool call:
  ```json
  {"t": 312.4, "turn": 41, "kind": "action", "step": "s4",
   "reasoning": "egg's done; turning off the burner before plating",
   "tool": "turn_off", "args": {"id": "burner_2"},
   "result": {"success": true, "error": null, "message": "burner_2 off"},
   "world_diff": ["- burner_front_left: on", "+ burner_front_left: off"],
   "plan_diff": [],
   "duration_s": 3.1, "request": "requests/0041.json"}
  ```
  `world_diff` = the GroundTruthObserver's relation lines before vs after the call (added `+`, removed `-`). `plan_diff` = plan changes caused by the call.
- **`event`**: timer/monitor/safety/motion events as delivered.
- **`check`**: from `complete()` or a monitor: step, condition, passed, reason, file in `checks/`.
- **`plan_edit`**: every live-plan change (also in the edit log).
- **`llm_error`** / **`reprompt`**: no tool call, invalid args, provider errors.
- **`limit`**, **`run_end`**.

### 11.3 Truth vs belief
Because `world_diff` comes from ground truth and `result` from the tool, the log shows **disagreements** directly: a `pick` that returned success with an empty `world_diff` (nothing became held), or a `complete()` that passed while ground truth says the condition is false. The viewer highlights these (§11.4). They are the main thing to look for when debugging.

### 11.4 Viewer

`python -m chefbot.scripts.view_run runs/<dir>` → writes `report.html` (one self-contained file) and prints a text summary.

**Text summary** (terminal):
```
fried_egg · finished · 58 turns · 6:40 sim · 4:12 wall · $0.31
plan: 6/6 done · 1 step added (s2a) · 0 skipped · 2 check failures · 1 replan request
mismatches: 1 (turn 17: pick success, nothing held)
```

**report.html** sections:
1. **Header:** task, outcome, model ids, totals, link to video.
2. **Start state:** reference plan, and initial ground truth as a short relation list.
3. **Timeline** (the main view). One row per action, grouped under the step that was active:
   ```
   s3 Cook the egg ................................................ waiting → done
     4:58  [event] monitor passed: egg whites set
     4:59  "egg's done, turn off the burner first"
           turn_off(burner_2) ✓ burner_2 off          world: burner_front_left on → off
     5:03  complete(s3) ✓ (judge: egg in pan, whites set)          plan: s3 waiting → done
   ```
   - Columns: time, reasoning (one line), call + result (✓/✗ + message), world diff, plan diff.
   - Errors in red; truth/belief mismatches flagged.
   - Each row expands to the full tool result, the judge input/output, and the full LLM request for that turn.
   - Filters: by step, errors only, mismatches only, hide perception calls.
   - If a video exists, clicking a row's time seeks the video.
4. **Plan over time:** a compact chart of step statuses vs time (shows parallel waiting steps and pauses).
5. **End state:** final ground truth, live plan vs reference plan (added, skipped, reordered steps highlighted), full edit log.

`view_run --text` prints the timeline as plain text (same rows, no expansion) for quick terminal use.

## 11.5 Ground-truth state in the log only
`start.json`, `end.json` and `world_diff` use the observer. That's fine because the log is for humans, but it must never be fed back into the agent context.

## 12. Running

```
python -m chefbot.scripts.run_agent --task tasks/fried_egg.yaml --layout 1 --style 1 \
    [--time-mode paused|realtime] [--model ...] [--record-video] [--remote ws://...]
```
Writes the run directory and generates the report at the end (also on crash/limit).

## 13. Test tasks (hand-written plans, `tasks/`)

Increasing difficulty. Each task file = goal, steps (goal, done_when, depends_on, hints), the scene spawn set, and optional **disturbances** (test hooks that change the world at a given time or step, e.g. move an object, occupy a burner).

| Task | What it tests |
|---|---|
| `mug_to_counter` | loop basics: one step, go_to → detect → align → pick → place → complete |
| `egg_to_pan` | fridge open/close, multi-step dependencies, check failures |
| `fried_egg` | burner on/off, timer + monitor, waiting step, plate in parallel, safety off |
| `fried_egg_pause` | as above with a long second task (move 3 items to counter_2); egg monitor fires mid-task → agent must pause and resume |
| `fried_egg_blocked` | disturbance: a pot occupies the planned burner → agent adds a step or picks another burner |
| `missing_ingredient` | no egg in the fridge → agent should `request_replan`, not improvise |

## 14. Acceptance criteria

**Unit (no sim, no LLM):**
- Plan store enforces every transition rule in §4.2 with the right error codes; edit log records every change; reference plan never changes.
- Event bus: timers fire at the right sim time; monitors call the checker at the right interval, stop on pass, emit timeout; `wait` returns on motion end / event / timeout / nothing_to_wait_for.
- Context builder: static prefix byte-identical across turns; only the newest task state present; rolling window size N; perception freshness transitions per §9.5.
- Loop with a scripted fake LLM: text-only turn → re-prompt; invalid args → `invalid_args`; second parallel call → `one_call_per_turn`; limits stop the run.
- Run log: every record kind written; `world_diff` correct on a fake observer; viewer renders a fixture log to HTML and text.
- Boundary: observer output never reaches the context builder.

**Sim, no LLM (scripted tool calls through the loop):**
- Cook model: egg in pan on lit burner reaches `set` at the configured time; off burner → no heating.
- Idle stepping in both time modes; `paused` mode fast-forwards during `wait`.
- Safety: empty lit burner turns itself off after the limit.

**Sim + LLM:**
- `mug_to_counter`, `egg_to_pan`, `fried_egg` complete on layouts 1 and 2 in ≥ 3/5 runs each (report rates, turns, cost).
- `fried_egg_pause`: agent pauses the second task, handles the egg, resumes; plan log shows pausing → paused → active.
- `missing_ingredient`: agent calls `request_replan`.
- Every run produces a report; a reviewer can find the first failing action from the text summary in under a minute.

## 15. Future work

- Planning agent: recipe search, grounding against the map and skills, generating plans in this format; `request_replan` calls it.
- Images in the agent context; VLM judge over camera images; proprioception/force in observations.
- Parallel tool calls for non-motion tools.
- Learned summarization of long histories (v0.1 relies on step logs + window).
- Real-robot time mode (wall clock only) and real burner/appliance control.
- More skills (stir, pour, crack) as tools, then π0.5 as a tool.
- Evaluation harness across many seeded runs with automatic mismatch statistics.

## 16. Open questions (answer before coding the affected part)

1. Can RoboCasa's stove burners/knobs be set and read? (§5.4)
2. How are "inside" and "on" best computed in RoboCasa (contacts vs bounding-box containment)? Pick one and document thresholds. (§5.2)
3. Which OpenRouter model/provider for the agent and for the judge? Confirm tool calling works with the chosen provider.
4. Does the existing rollout recorder work alongside the agent loop in one process (macOS GL context limit noted in the project spec)?
