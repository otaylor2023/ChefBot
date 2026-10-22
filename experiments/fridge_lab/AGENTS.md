# Fridge lab: research brief

You are a robotics researcher. Your job tonight is to write a **skill**: a Python
program that makes a mobile manipulator open a fridge door. ("Skill" here always means
this robot program, not a Codex skill.) It should work on fridges
you have never seen, and it should be **fast and cheap**.

## Why

A large VLM can open these doors as the robot's policy, choosing every motion itself.
This was measured on one fridge, with a mobile Franka:

| Policy | Success at 60° / 85° | Time per episode | Cost per episode |
|---|---|---|---|
| GPT-6 Astra, one API call per action chunk ("Direct") | 4/5 / 0/1 | ~10 min | ~$2.50 |
| GPT-6 Astra as a tool-using agent | 5/5 / 5/5 | 5–16 min | (subscription) |

Your skill must also work across many fridge models, and do so in well under a minute
for well under a cent. You may call small models while the skill runs (see Models), but
the skill pays for every second and every cent it spends.

## The robot and the task

{ROBOT}
- **Control.** 11-D whole-body targets at 15 Hz. The full contract is in
  `docs/fridge_robot.py`: read it first.
- **Physics.** The door is a real hinged body. It opens only if the robot pulls it
  open, usually by grasping the handle and moving along the door's arc.
  - Doors are 30–50 kg with damping.
  - Handles differ between models: long vertical bars, short bars, and bars on
    mounts.
  - Fridge types: French door (two doors; either one counts), side-by-side (the
    fridge door), and bottom freezer (the top door).
  - Kitchens can be narrow, with cabinets or an island close behind the robot.
- **Success.** The door's final angle is at least 60° (also reported at 85°; aim for
  fully open). The skill decides when it is done: return from `run(robot)`.

{PRIOR}## Scoring

Your final skill is scored by us on a **held-out test set: fridge models and kitchens
you never see**. Ranked by:

1. Success rate at 60° and at 85°.
2. **Time** per episode: wall time in the world. **The simulation never pauses for
   you:** while your skill computes or waits on a model, the world runs on, with the
   robot holding its last command. An episode has 90 s of world time, thinking included.
   Overlapping computation with motion (a thread while `act()` runs) is fine and is
   free, as on a real robot.
3. Model dollars per episode.

**Per-episode limits:** {LIMITS}. That is control steps, think seconds, model USD, and
a hard wall-clock cap.

## Models the skill may call

All model calls go through the robot object, metered.

- **`robot.llm(...)`: Gemini.** Prices per million tokens are listed in
  `robot.models()`. One call with an image costs about $0.0003 (3.1 Flash-Lite) to
  $0.0025 (Robotics-ER) and takes 1–3 s. The total for all your dev and val episodes
  tonight is **${MODEL_USD}**. Each episode is capped as above.
- **`robot.detect(...)`.** Open-vocabulary boxes from OWLv2 or Grounding DINO, on a
  GPU. No dollar cost, but each call costs time (about 0.1–0.5 s).
- **`robot.segment(...)`.** SAM 2 masks from boxes or points, on a GPU. Also free in
  dollars and costly in time.
- Depth plus intrinsics plus camera poses gives you 3-D points. That needs no model.

The skill runs in a sandbox: no network, no subprocesses, no files outside its own
directory, and no simulator. Its Python has numpy, scipy, opencv (headless) and pillow.
Sandbox here: {SANDBOX}. Blocked attempts are logged and counted in every result.

## Your tools

`./lab` (run `./lab -h`):

- `./lab scenes`: the **dev** scenes (fridge doors, their type and model).
- `./lab run skills/X --scene dev05 --seed 0`: one episode.
  - Results go to `runs/<job>/<scene>_s<seed>/`:
    - `result.json`;
    - `video.mp4` and `keyframes.jpg`;
    - `skill_log.txt` (your `robot.log` lines and prints);
    - `images/` (from `robot.save_image`);
    - `calls.jsonl`;
    - `trace.jsonl`.
  - **`trace.jsonl`** has the per-step door angles, the true handle position, and what
    the robot touched. It is ground truth, for *your* debugging only. The skill can
    never see it, so don't build a skill that depends on it.
- `./lab suite skills/X --scenes all --seeds 0,1 --no-video`: a batch in the
  background (episodes run in parallel). Then `./lab wait <job>`.
- `./lab val skills/X`: success and time on the **held-out val scenes**, as an
  aggregate only. **{VAL_SUBMISSIONS} submissions for the whole night.** Use them to
  check generalization, not to tune.
- `./lab budget`: model dollars spent, val submissions left, time left, and your own
  usage.

You can look at images and videos. Open the keyframes and saved images to see what
went wrong.

## Rules

This is research, and the result is only worth something if it's honest. Please keep
to these, and if something is ambiguous, write it down in `research_log.md` and make
the conservative choice.

1. **Don't look for the val or test sets.**
   - Don't open or search for RoboCasa's kitchens, layouts, styles or fridge models, the
     lab's private directory, the scene files (`scenes*.json`), or any val or test
     results beyond what `./lab val` prints.
   - Don't build simulator environments yourself. Use `./lab` only.
{WEB}2. **The skill uses the robot API only.**
   - No simulator state, no files from runs, no hidden channels.
   - Don't try to get around the sandbox, the budgets or the time accounting.
3. **No lookup tables of specific fridges.**
   - Don't hard-code per-model or per-scene geometry, poses or IDs. The skill must work
     on fridges it has never seen.
   - General priors are fine, for example "handles are within 15 cm of the door edge"
     or "doors open about a vertical hinge".
4. **Don't modify the lab.** That includes `./lab`, `docs/`, and the code it runs.
   Reading the lab's source in the ChefBot repo to understand the controller or the
   time accounting is fine.
5. **Report honestly.** That includes failures, negative results, and anything you
   did that might be borderline.

## How to work (the loop)

1. **Start small.**
   - Read `docs/fridge_robot.py` and `skills/example/`.
   - Run the example on a couple of dev scenes and look at the outputs.
2. **Iterate.**
   - Pick the biggest failure.
   - Form a hypothesis and make one change, in a new skill version (`skills/v2`,
     `skills/v3`, ...).
   - Compare against the previous version on the **same** dev scenes and seeds.
   - Keep it or reject it based on the numbers, not the story.
   - Look at the videos and keyframes of failures.
3. **Write each experiment up** in `research_log.md`:
   - the hypothesis and the change;
   - the dev result versus the previous best;
   - whether you kept it or not.
4. **Keep every version** as its own `skills/vN` folder. The lab snapshots each one you
   run. Commit with git if your sandbox allows it; if it doesn't, the folders are the
   record.
5. **Use val sparingly.** Use it when a version is clearly better on dev, or to
   compare two very different approaches.
6. **Make it robust, then fast, then cheap.**
   - A skill that opens the door 90% of the time in 30 s beats one that opens it 60%
     of the time in 15 s.
   - Then cut time: fewer and smaller model calls, fewer observations, faster motions,
     doing work while moving.

Useful things to think about:
- finding the handle robustly across handle styles;
- grasping it so that it doesn't slip;
- inferring the hinge side and the door's arc from what you see;
- moving the base as the door swings, and staying clear of the door and the cabinets
  behind you;
- detecting failure (empty gripper, door not moving) and recovering;
- knowing when the door is open.

**Session.** About {HOURS} hours of wall time. Codex usage is capped at {USAGE_CAP}%
of the weekly limit. When less than about 40 minutes are left, `./lab` responses tell
you to wrap up.

**Wrap-up (required).**
- Put your best skill in `skills/final`, and run it once with `./lab` to make sure it
  works.
- Write `REPORT.md` with:
  - your approach;
  - what worked and what didn't (with numbers);
  - the dev and val results of the final skill;
  - its expected time and cost per episode;
  - its known failure modes;
  - whether it could transfer to a real robot.
