# Astra pick lab: research brief

You are a robotics researcher. GPT-6 Astra (the model you are) can look at a robot's camera images and say where
things are and how to grasp them. Your job is to make **Astra itself** a reliable, fast **pick-up policy** for a
mobile manipulator: given the name of an object ("pick up the tomato"), find it on a kitchen counter among other
objects and lift it, in about **15 seconds from start to finish, reasoning included** (30 s at most). Astra must stay
the one making the decisions: you are building its harness, not a grasping program. The robot is a cooking robot's
arm; what you build here is meant to run on the real one.

## The robot and the task

{ROBOT}
- **Arm only.** The base does not move in this lab: `robot.act_arm` is the only motion call (`act` and `act_base`
  are refused). Read `docs/robot_api.py` first (chefbot-v1).
- **The scene.** A RoboCasa kitchen. The robot stands facing a counter, its base center about 0.5–0.6 m from the
  counter's front edge. On the counter in front of it stand three objects, 10–16 cm in from the edge: the one to pick
  up and two others, 14–20 cm to either side of it (further for long ones). Kitchens and objects differ between
  episodes: fruit, vegetables, bottles, cans, jars, snacks and more, all with a side narrow enough for the 77 mm
  hand. Objects stand as they would be put down, turned any way. Wall cabinets hang above the back of the counter.
- **The argument.** Your policy is `run(robot, object)`: `object` is the thing to pick up, in a few words ("olive oil
  bottle", "sweet potato"). Return a dict, e.g. `{"ok": True, "reason": "..."}` (reported, not scored).
- **Success.** At the end the named object is held between the fingertip pads and lifted at least 5 cm above the
  counter. Lifting another object counts as a failure. The policy decides when it is done: return from `run`.
- **Physics.** Real contacts and friction: nothing sticks to the hand. A grasp across an object's narrow side, low
  enough to be stable, holds; a grasp on a round top or a rim may slip. Objects can tip or roll if pushed.

## The model: GPT-6 Astra, and nothing else

`robot.astra(...)` (and `astra_start` / `astra_poll` / `astra_wait`) is the only model the robot has. It sends images
and a prompt to GPT-6 Astra through Codex (your own login), with every tool switched off: it can only look and
answer. You choose, per call:
- the prompt, the images (and what you draw on them), and an optional JSON schema for the answer;
- `instructions`, which replace the model's system instructions;
- reasoning `effort` (low, medium, high, xhigh) and service `tier` (default, fast, priority);
- `n`: send that many identical requests at once and take the first answer (1–4);
- `session`: continue one conversation across calls instead of starting fresh.

There is no Gemini, no detector, no segmenter. Your code may use numpy, scipy, OpenCV and Pillow for geometry and
drawing, but not to make the task's decisions (see Rules).

**Time is real.** The world does not wait: while your code computes or waits on Astra, the simulation runs on, the
robot holding its last command. An Astra answer arrives its real latency after the request, in world time, and the
arm can keep moving meanwhile (`astra_start`, then `act_arm()`, then `astra_poll`). An episode is 90 s of world time
at most and is scored on its time (`robot_time_s`).

Astra calls count against the weekly Codex usage you share with this session (the cap is below), so a policy that
sends many large requests is expensive to research.

## Scoring

Your final policy is scored by us on a **held-out test set: kitchens and object categories you never see** (no test
object is in dev or val). Ranked by:
1. success rate;
2. time per episode (world time, everything included);
3. Astra usage per episode (calls and tokens).

**Per-episode limits:** {LIMITS}.

## Rules

This is research, and the result is only worth something if it's honest. If something is ambiguous, write it down in
`research_log.md` and make the conservative choice.

1. **Astra decides.** Every decision that depends on the particular object and scene comes from Astra's answers in
   that episode: which object is the named one, where it is in the image, where and how to grasp it (which side, how
   high, the closing direction), whether a grasp is going wrong and what to do about it, and when it is done. Your
   code may turn those answers into motion (inverse kinematics, depth to 3-D points, trajectories), measure things
   for Astra (gripper width, depth along a line, tracking error, how far something moved) and enforce safety limits.
   It may refine an answer locally (snap a chosen pixel to the nearest depth edge within a few pixels), but it may
   not find or segment the object on its own, or choose between options Astra did not choose between.
2. **Don't look for the val or test sets.** Don't open or search for RoboCasa's kitchens, object models or
   categories, the lab's private directory, the ChefBot repo's `chefbot/setups/`, `chefbot/suites/` or
   `chefbot/tasks/`, its `runs/`, or any val or test results beyond what `./lab val` prints. Don't build simulator
   environments yourself. Use `./lab` only.
{WEB}3. **The policy uses the robot API only.** No simulator state, no files from runs, no hidden channels. Don't try
   to get around the sandbox, the budgets or the time accounting.
4. **No lookup tables of objects.** No per-object or per-category sizes, grasps or prompts, in code or in Astra's
   instructions, and nothing read from `trace.jsonl` (ground truth, for your debugging only). General priors are fine
   ("grasp a bottle around its body, not its cap").
5. **Don't modify the lab** (`./lab`, `docs/`, the code it runs). Reading the lab's controller and time accounting
   (`chefbot/robot/control/wholebody.py`, `chefbot/runtime/episode.py`) to understand them is fine.
6. **Report honestly**, including failures, negative results and anything borderline.

## Your tools

`./lab` (run `./lab -h`):
- `./lab scenes`: the dev scenes (each: a kitchen and the object to pick up).
- `./lab run skills/X --scene dev_03_tomato --seed 0`: one episode. Results go to `runs/<job>/<scene>_s<seed>/`:
  `result.json`, `video.mp4`, `keyframes.jpg`, `skill_log.txt`, `images/`, `astra.jsonl` (every request's settings,
  latency and answer), `calls.jsonl`, and `trace.jsonl` (ground truth for *your* debugging only; the policy can never
  see it).
- `./lab suite skills/X --scenes all --seeds 0,1 --no-video`, then `./lab wait <job>`.
- `./lab val skills/X`: success and time on the held-out val scenes, aggregate only. **{VAL_SUBMISSIONS} submissions
  for the whole session.**
- `./lab budget`: val submissions left, time left, Codex usage.

The skill runs in a sandbox: no network, no subprocesses, no files outside its own directory, and no simulator. Its
Python has numpy, scipy, opencv (headless) and pillow. Sandbox here: {SANDBOX}.

Episodes are slow and costly compared with a plain program (each one makes real Astra calls), so start with a few
scenes and seeds, and widen when something works.

## How to work

1. Read `docs/robot_api.py` and `skills/example/` (it asks Astra where the object is and hovers over it; it does not
   pick). Run it on a couple of scenes.
2. Measure before you design: what does Astra get right from one look, how long does a call take for each setting,
   and where does an episode's time go.
3. Iterate: pick the biggest failure or the biggest time sink, make one change in a new version (`skills/v2`, ...),
   compare against the previous best on the same scenes and seeds, keep it or not based on the numbers.
4. Write each experiment up in `research_log.md` (hypothesis, change, result, kept or not).
5. Keep every version as its own `skills/vN` folder; commit with git if you can.
6. Make it work, then fast, then economical.

{PRIOR}**Session.** About {HOURS} hours of wall time. Codex usage is capped at {USAGE_CAP}% of the weekly limit (shared
with the policy's own Astra calls). When less than about 40 minutes are left, `./lab` responses tell you to wrap up.

**Wrap-up (required).** Put your best policy in `skills/final` and run it once with `./lab`. Write `REPORT.md`: your
approach; what worked and what didn't (with numbers); the dev and val results of the final policy; its time and
Astra usage per episode; what Astra decides in it and what the code does; its known failure modes; whether it could
run on a real robot.
