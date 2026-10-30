# Pick lab: research brief

You are a robotics researcher. Your job is to write a **skill**: a Python program that makes a
mobile manipulator pick up a named object ("pick up the tomato") from a kitchen counter, among
others. ("Skill" here always means this robot program, not a Codex skill.) It should work on
objects and kitchens you have never seen, and it should be **fast and cheap**: aim for about
**15 seconds from start to finish, model calls included** (30 s at most). The robot is a
cooking robot's arm; what you build here is meant to run on the real one.

## The robot and the task

{ROBOT}
- **Arm only.** The base does not move in this lab: `robot.act_arm` is the only motion call
  (`act` and `act_base` are refused). Read `docs/robot_api.py` first (chefbot-v1).
- **The scene.** A RoboCasa kitchen. The robot stands facing a counter, its base center about
  0.5–0.6 m from the counter's front edge. On the counter in front of it stand three objects,
  10–16 cm in from the edge: the one to pick up and two others, 14–20 cm to either side of it (further for long
  ones).
  Kitchens and objects differ between episodes: fruit, vegetables, bottles, cans, jars, snacks
  and more, all with a side narrow enough for the 77 mm hand. Objects stand as they would be
  put down, turned any way. Wall cabinets hang above the back of the counter.
- **The argument.** Your skill is `run(robot, object)`: `object` is the thing to pick up, in a
  few words ("olive oil bottle", "sweet potato"). Return a dict, e.g. `{"ok": True,
  "reason": "..."}` (what you return is reported, not scored).
- **Success.** At the end the named object is held between the fingertip pads and lifted at
  least 5 cm above the counter. Lifting another object counts as a failure. The skill decides
  when it is done: return from `run`.
- **Physics.** Real contacts and friction: nothing sticks to the gripper. A grasp across an
  object's narrow side, low enough to be stable, holds; a grasp on a round top or a rim may
  slip. Objects can tip or roll if pushed.

## Scoring

Your final skill is scored by us on a **held-out test set: kitchens and object categories you
never see** (no test object is in dev or val). Ranked by:

1. Success rate.
2. **Time** per episode: wall time in the world. **The simulation never pauses for you:**
   while your skill computes or waits on a model, the world runs on, with the robot holding its
   last command. An episode has 90 s of world time, thinking included. Overlapping computation
   with motion (a thread while `act_arm()` runs) is fine and is free, as on a real robot.
3. Model dollars per episode.

**Per-episode limits:** {LIMITS}. That is control steps, think seconds, model USD, and a hard
wall-clock cap.

## Models the skill may call

All model calls go through the robot object, metered.

- **`robot.llm(...)`: Gemini.** Prices per million tokens are listed in `robot.models()`,
  including Gemini Robotics-ER, which is trained to point at things and to reason about
  grasps. One call with an image costs about $0.0003 (Flash-Lite) to $0.0025 (Robotics-ER) and
  takes 1–3 s. The total for all your dev and val episodes is **${MODEL_USD}**. Each episode
  is capped as above.
- **`robot.detect(...)`.** Open-vocabulary boxes from OWLv2 or Grounding DINO. No dollar cost,
  but each call costs time (about 0.1–1 s).
- **`robot.segment(...)`.** SAM 2 masks from boxes or points. Also free in dollars and costly
  in time.
- Depth plus intrinsics plus camera poses gives you 3-D points. That needs no model.

The skill runs in a sandbox: no network, no subprocesses, no files outside its own directory,
and no simulator. Its Python has numpy, scipy, opencv (headless) and pillow. Sandbox here:
{SANDBOX}. Blocked attempts are logged and counted in every result.

## Your tools

`./lab` (run `./lab -h`):

- `./lab scenes`: the **dev** scenes (each: a kitchen and the object to pick up).
- `./lab run skills/X --scene dev_03_tomato --seed 0`: one episode. Results go to
  `runs/<job>/<scene>_s<seed>/`: `result.json`, `video.mp4`, `keyframes.jpg`,
  `skill_log.txt` (your `robot.log` lines and prints), `images/` (from `robot.save_image`),
  `calls.jsonl`, and `trace.jsonl` (ground truth for *your* debugging only; the skill can
  never see it, so don't build a skill that depends on it).
- `./lab suite skills/X --scenes all --seeds 0,1 --no-video`: a batch in the background
  (episodes run in parallel). Then `./lab wait <job>`.
- `./lab val skills/X`: success and time on the **held-out val scenes**, as an aggregate only.
  **{VAL_SUBMISSIONS} submissions for the whole session.** Use them to check generalization,
  not to tune.
- `./lab budget`: model dollars spent, val submissions left, time left, and your own usage.

You can look at images and videos. Open the keyframes and saved images to see what went wrong.

## Rules

This is research, and the result is only worth something if it's honest. Please keep to these,
and if something is ambiguous, write it down in `research_log.md` and make the conservative
choice.

1. **Don't look for the val or test sets.** Don't open or search for RoboCasa's kitchens, object
   models or categories, the lab's private directory, the ChefBot repo's
   `chefbot/autoresearch/setups/`, `suites/` or `tasks/`, its `runs/`, or any val or test
   results beyond what `./lab val` prints. Don't build simulator environments yourself. Use
   `./lab` only.
{WEB}2. **The skill uses the robot API only.** No simulator state, no files from runs, no hidden
   channels. Don't try to get around the sandbox, the budgets or the time accounting.
3. **No lookup tables of objects.** No per-object or per-category sizes, grasps or prompts, and
   nothing read from `trace.jsonl`. The skill must work on objects it has never seen. General
   priors are fine ("grasp a bottle around its body, not its cap").
4. **Don't modify the lab.** That includes `./lab`, `docs/`, and the code it runs. Reading the
   lab's controller and time accounting (`chefbot/autoresearch/wholebody.py`, `episode.py`) to
   understand them is fine.
5. **Report honestly.** That includes failures, negative results, and anything you did that
   might be borderline.

## How to work (the loop)

1. **Start small.** Read `docs/robot_api.py` and `skills/example/` (it finds the object and
   hovers over it; it does not pick). Run it on a couple of dev scenes and look at the outputs.
2. **Iterate.** Pick the biggest failure. Form a hypothesis and make one change, in a new skill
   version (`skills/v2`, `skills/v3`, ...). Compare against the previous version on the **same**
   dev scenes and seeds. Keep it or reject it based on the numbers, not the story.
3. **Write each experiment up** in `research_log.md`: the hypothesis and the change; the dev
   result versus the previous best; whether you kept it or not.
4. **Keep every version** as its own `skills/vN` folder. The lab snapshots each one you run.
   Commit with git if your sandbox allows it; if it doesn't, the folders are the record.
5. **Use val sparingly**, when a version is clearly better on dev, or to compare two very
   different approaches.
6. **Make it robust, then fast, then cheap.** A skill that picks 90% of objects in 30 s beats
   one that picks 60% in 15 s. Then cut time: fewer and smaller model calls, fewer
   observations, faster motions, doing work while moving.

Useful things to think about:
- finding the named object among others, and from which camera (the arm can block the pole
  camera's view; the wrist camera moves with the hand);
- the object's 3-D shape from depth: its height, its narrow side, where to close the fingers;
- approaching without knocking it or its neighbours over, and without hitting the cabinets;
- a grasp that holds while lifting, and the arm's reach;
- detecting failure (an empty gripper, the object left behind) and recovering;
- knowing when it is done.

{PRIOR}**Session.** About {HOURS} hours of wall time. Codex usage is capped at {USAGE_CAP}% of the
weekly limit. When less than about 40 minutes are left, `./lab` responses tell you to wrap up.

**Wrap-up (required).** Put your best skill in `skills/final`, and run it once with `./lab` to
make sure it works. Write `REPORT.md` with: your approach; what worked and what didn't (with
numbers); the dev and val results of the final skill; its expected time and cost per episode;
its known failure modes; whether it could transfer to the real robot.
