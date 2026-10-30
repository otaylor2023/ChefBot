# Elevator lab: research brief

You are a researcher building simulation environments for robot learning. Your job is to write a **generator**: a
Python program that, given a seed, makes **one building with elevators** as a MuJoCo scene, accurate in how it looks
and in how it behaves, so that many seeds give a realistic **variety** of real buildings and their elevators. A
mobile robot (a Franka arm on a wheeled base) will later learn in these buildings to call an elevator, wait, get in
over the sill, press its floor, ride, and get out; what it learns should carry over to real elevators it has never
seen. So the buildings must look like real ones (cameras see them), behave like real ones (waits, doors, rides,
displays, and their quirks), and be physical where a robot touches them (sills, gaps, door edges, buttons).

## What to make

- **The building**: floors (up to 60), halls and lobbies, shafts, cars (single cars and banks), doors (centre- and
  side-opening, panel counts, speeds), call panels, car panels, Braille and signage, hall lanterns and position
  indicators, screens and destination-dispatch keypads, finishes. Building types (office, residential, hotel,
  hospital, parking, ...) and their elevators differ: decide which to cover and how often.
- **The controller** (the scene's `dynamics.py`): everything about how the elevators behave. Which presses count,
  dispatch (collective, destination dispatch, banks), motion profiles and leveling, door timing, nudging and
  reopening, lamps and screens, background traffic, and **faults**: real elevators misbehave sometimes (a press
  that does not register, a button that needs holding, a slow or out-of-service car, a door that reopens). Faults
  should be rare by default, each with a rate parameter and a way a person would recover.
- **Parameters**: build parameters (what needs a new building) and run parameters (what the controller reads and can
  change per episode: timings, fault rates, traffic), all in physical units, with ranges you set and **justify** in
  ranges.yaml from codes (ASME A17.1, ADA, EN 81-70), manufacturer catalogs, measurements and studies.
  `generate(seed, out_dir, overrides)` must honour overrides: we will ask for specific buildings (e.g. 20 floors, two
  cars) and specific run parameters.
- **The contract** (`docs/generator_contract.md`): the scene format, the annotations (what everything is), the
  controller's interface. Read it first.

## How it is scored

On fresh seeds, after the session, by us, in this order:

1. **Pass rate**: the share of buildings that pass every check in `docs/checks.md`. The checks run your building
   with the lab's probes: a fingertip presses buttons, a small wheeled cart takes a full trip (call, wait, get in,
   press, ride, get out), a block tests the doors, and more. They compare what happens with what you declare.
2. **Realism**: a judge (Claude, a different model from you) scores each building's renders and trip against
   photos of **real elevators you have not seen**. Its dev rubric is `docs/judge_rubric.md`; val and test use a
   held-out rubric and other photos. People will also look at the renders and the trip videos.
3. **Variety**: across a batch, how much of the range of real buildings it covers (`docs/judge_rubric_batch.md`).
4. Speed: generation (keep it under a minute or two per building) and simulation (it must stay faster than real
   time; a check measures it).

`refs/` holds the dev photos of real elevators (`refs/real_elevators.jpg` is what the dev judge sees). You have the
internet too.{WEB}

## Your tools

`./lab` (run `./lab -h`):

- `./lab build generators/X --seeds 0-3 --wait`: generate, check and render (at most {MAX_SEEDS} seeds per build).
  Results in `runs/<job>/`: per seed, the scene itself and `lab/` with `checks.json` (every scenario's numbers and
  why it failed), `sheet.jpg` (six views), `actuation.jpg` (the trip in frames), `actuation.mp4` (a time-lapse of the
  trip: origin hall, car, destination hall); and `batch.jpg`. **Look at the images and videos.**
- `./lab judge <job> [--seeds 0,3]`: the dev judge on a finished build: scores and the flaws it sees per building,
  and the batch's variety. One judge call per building, plus one for the batch; **{DEV_JUDGE_CALLS} calls for the
  whole session** (`./lab budget`).
- `./lab val generators/X`: {VAL_N} fresh seeds, the held-out rubric, a stronger judge, other photos: **aggregate
  scores only. {VAL_SUBMISSIONS} submissions for the session.**
- `./lab budget`: judge calls, val submissions and time left.

Tall buildings take longer to check (a 60-floor building's full checks take minutes): build a few seeds while you
iterate, and widen when something works. The generator and your controller run in sandboxes: no network, no files
outside their own directory and the output directory, {SANDBOX}. Their Python ({GEN_PYTHON}) has the packages the
contract lists; you can run the generator and the environment yourself too (`chefbot/environments/mujoco`). If a
package you need is missing, say so in `research_log.md`; we can install it between sessions.

## Rules

This is research, and the result is only worth something if it's honest. If something is ambiguous, write it down
in `research_log.md` and make the conservative choice.

1. **Don't look for the val or test photos or the held-out rubric**, or the lab's private files, or other runs. Use
   `./lab` for builds and judging.
2. **The generator makes the buildings from parameters.** Downloaded models or photos of particular buildings can't
   be the output (or be traced into it). Real dimensions, codes, timings and designs learned from the web are
   exactly what to use. Textures you make procedurally are fine.
3. **No gaming the checks or the judge**: a controller that behaves differently when it detects the lab's probes, a
   building that only looks right from the lab's cameras, text in the renders addressed to the judge, or parameters
   declared to match measurements you did not model.
4. **Don't modify the lab**: `./lab`, `docs/`, and the lab code (`chefbot/generators/elevator/`,
   `chefbot/autoresearch/`, `chefbot/environments/mujoco/`). Reading it is fine.
5. **Report honestly**, including failures, negative results, and anything borderline.

## How to work

1. **Start small.** Read the contract, `docs/checks.md` and `generators/example/` (one car, boxes, a simple
   controller; it passes the checks at any height up to 60 floors). Build it, and look at the sheets and videos.
2. **Research real elevators**: building types and their installations, codes, dimensions, door and panel designs,
   controllers and dispatch, timings, common faults. Write down what you find and where from.
3. **Iterate.** One change at a time, in new versions (`generators/v2`, `v3`, ...). Compare on the **same** seeds.
4. **Write each experiment up** in `research_log.md`: what you changed, why, what happened.
5. **Use the judge and val sparingly**, when you think a version is clearly better.
6. **Working first, then looks, then variety.** A building that fails a check is worth nothing to the robot.

**Session.** About {HOURS} hours of wall time. Codex usage is capped at {USAGE_CAP}% of the weekly limit. When less
than about 40 minutes are left, `./lab` responses tell you to wrap up.

**Wrap-up (required).** Put your best generator in `generators/final` (with its ranges.yaml), and build it once with
`./lab` to make sure it works. Write `REPORT.md` with: your approach; the kinds of buildings and controllers it makes
and their parameters; what worked and what didn't (with numbers); the dev and val results; known flaws; what you'd
do next.
