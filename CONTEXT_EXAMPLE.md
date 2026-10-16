# ChefBot agent — example context window (one turn)

Scenario: fried egg on a plate. The egg is cooking; the agent is fetching a plate when the egg monitor fires mid-drive.

Assumes two tools that don't exist yet: `turn_on` / `turn_off` (burners; scripted like `open`), and the plan/timer tools from the agent-loop design. Everything else matches the current ToolRegistry.

Order matters for prompt caching: **[1] system and [2] tools never change within a run**; [3] is the rolling window; [4] is re-rendered every turn and always last. Only the newest [4] is kept — older state messages are dropped from history so they don't pile up.

---

## [1] System prompt (static, cached)

```
You control a mobile robot (wheeled base + one arm with a parallel gripper) in a kitchen.
You act only through tools. Each turn you receive the current TASK STATE at the end of the
conversation; trust it over anything older.

HOW TO ACT
- Get somewhere: go_to(place) with a name from MAP PLACES. This is coarse: it gets the
  place in view, not in reach.
- See what's there: detect(classes) for the things you need. Map places near you are
  checked automatically; nearby() shows everything around you.
- Act on something: only ids listed as VISIBLE in the latest nearby/detect results can be
  used with align, pick, place, open, close, turn_on, turn_off. Places listed as "nearby,
  not visible" need go_to first.
- Before pick/place/open on something, align(id) so the arm can reach it. If a tool
  returns `unreachable`, align and retry.
- Typical sequence: go_to -> detect -> align -> pick -> go_to -> detect -> align -> place.
- If a tool fails, read the error and message; they say what to do next. Don't repeat
  the same failing call unchanged more than twice.

PERCEPTION FRESHNESS
- Detections are kept until you move far (then cleared). After a small move or several
  tool calls they are marked "possibly stale": still usable, but re-detect if an action
  fails or before acting on something that may have moved.

MOTION AND WAITING
- Motion tools block until done, but you are woken early if an event arrives (the result
  then has status "moving" and the event). You can then stop(), or wait() to let the motion
  finish.
- When nothing needs doing, call wait(); it returns on the next event or when motion ends.
  Don't poll with repeated status calls.

THE PLAN
- TASK STATE shows the plan. Step statuses:
  pending (not started), active (you are working on it), pausing (you are winding it down,
  e.g. putting a tool down), paused (started, stopped; has a resume note), waiting
  (progressing without you, e.g. cooking), done, failed, skipped.
- Only one step can be active or pausing at a time. Many can be waiting.
- To finish a step call complete(step_id). It checks the step's done_when condition and
  marks it done only if the check passes; otherwise it tells you what's missing.
- To switch tasks mid-step: set_status(step, "pausing"), wind it down safely (put down
  what you're holding if needed), then set_status(step, "paused", note=<how to resume>).
- For things that take time, set_timer and/or watch(step) when you start the step, mark the
  step waiting, and go do something else. You'll get an event when it fires.
- You may add_step for things the plan missed (e.g. clearing a blocked burner) and
  skip_step with a reason. If the recipe itself can't be done (missing ingredient, wrong
  dish), call request_replan(reason) instead of improvising a different dish.

SAFETY
- Never leave a burner on with nothing on it. Turn burners off when a cooking step is done.
- The robot stops itself on collisions and safety events; the tool result will say so.
```

## [2] Tool schemas (static, cached) — abbreviated

```
go_to(place)                    detect(classes?)            nearby()
align(id)                       pick(id)                    place(id)
open(id) / close(id)            turn_on(id) / turn_off(id)  holding()
wait(timeout_s?)                get_status()                stop()
complete(step_id)               set_status(step_id, status, note?)
add_step(after, goal, done_when, reason)                    skip_step(step_id, reason)
set_timer(name, seconds, step_id?)                          cancel_timer(name)
watch(step_id, every_s, timeout_s)                          request_replan(reason)
```
(Full JSON schemas with descriptions and error codes in the real request.)

## [3] Rolling window — last N tool calls (N = 8 here)

```
assistant: set_status("s3", "waiting", note="egg in pan on lit burner_2")
tool:      ok · s3 waiting

assistant: set_timer("egg", 150, step_id="s3")
tool:      ok · timer 'egg' 2:30

assistant: watch("s3", every_s=20, timeout_s=300)
tool:      ok · monitoring s3 every 20s (judge: "egg whites set")

assistant: set_status("s4", "active")
tool:      ok · s4 active

assistant: go_to("cabinet_2")
tool:      success · arrived at cabinet_2

assistant: detect(["plate"])
tool:      success · Found: plate_1.

assistant: align("cabinet_2 handle_1")      # opened cabinet first
tool:      success
assistant: open("cabinet_2 handle_1")
tool:      success · cabinet_2 open (scripted)

assistant: align("plate_1")
tool:      success
assistant: pick("plate_1")
tool:      success · holding plate_1 (lift verified)

assistant: go_to("counter_1")
tool:      moving · woken by event: monitor s3 passed ("egg whites set")
```

## [4] Task state (re-rendered every turn, always last)

```
TASK STATE

GOAL: fried egg, served on a plate

PLAN
  s1  done      Get an egg from the fridge
                  log: egg_1 picked from fridge shelf after opening fridge; fridge closed
  s2  done      Pan on a burner, burner on
                  log: pan_1 placed on burner_2 (burner_1 had a pot, added s2a); burner_2 on
  s2a done      Move pot off burner_1   [added: pot was occupying burner_1]
                  log: actually not needed, used burner_2 instead
  s3  waiting   Cook egg in the pan until the whites are set
                  done_when: egg is in the pan on a lit burner and the whites are set
                  cooking 2:10 · timer 'egg' 0:20 left · monitor: PASSED (see event)
  s4  active    Get a plate onto counter_1
                  done_when: a plate is on counter_1
  s5  pending   Plate the egg              depends: s3, s4
                  done_when: egg is on the plate
  s6  pending   Turn off burner_2          depends: s5
                  done_when: burner_2 is off

NOW
  holding: plate_1
  motion: go_to("counter_1") in progress (about halfway)

PERCEPTION  (latest: detect(["plate"]) at cabinet_2 · possibly stale: moved since)
  visible:      plate_1 (held), cabinet_2, cabinet_2 handle_1, counter_2
  not visible:  stove, fridge

MAP PLACES
  fridge, stove, counter_1, counter_2, cabinet_1, cabinet_2, sink, microwave, toaster

EVENTS SINCE LAST TURN
  [monitor] s3 passed: "egg is in pan_1 on burner_2 (on); egg whites set"
  (timer 'egg' still running, 0:20 left)
```

---

## What the agent would plausibly do next

Egg is done; leaving it risks overcooking. It's holding a plate mid-drive, which is what s5 needs anyway, so a good agent:

1. `stop()` → then `go_to("stove")` instead of counter_1 (bring the plate to the egg), **or** lets the go_to finish, places the plate on counter_1 (completing s4), then goes to the stove.
2. `cancel_timer("egg")`.
3. Either way, `turn_off("burner_2")` early so the egg stops cooking.

Both are reasonable; the plan doesn't dictate it. That's the point of keeping the plan at subgoal level.

## Perception freshness rules used above

| Situation | Block shown as |
|---|---|
| Just detected, robot hasn't moved | fresh |
| More than K tool calls since detect (K≈5) | possibly stale (N calls ago) |
| Base moved slightly (e.g. `align`, `nudge`: > 5 cm or > 5°, < 0.5 m and < 30°) | possibly stale (moved since) |
| Base moved far (> 0.5 m or > 30°, usually `go_to`) | cleared — call detect/nearby |
| New detect / nearby | replaced |

Above, the robot is mid-`go_to` from cabinet_2, so the block is marked stale; it clears once the move passes 0.5 m from where the detect ran.
