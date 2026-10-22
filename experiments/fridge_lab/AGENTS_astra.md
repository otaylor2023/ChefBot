# Astra lab: research brief

You are a robotics researcher. GPT-6 Astra (the model you are) can open a fridge door with
a mobile manipulator when it drives the robot itself: given camera images and a simple
motion interface, it decides what to do, step by step, and nearly always succeeds. But it
takes minutes per door, because every small step waits on a model call.

Your job tonight is to make **Astra itself** a fast policy for this task: find the way of
running it (what it is shown, what it is asked, what it may answer, how its answers are
turned into motion, and when and how it is called) so that it opens the door in about
**20 seconds from start to finish, reasoning included** (30 s at most). Astra must stay the
one making the decisions. You are building its harness, not a fridge-opening program.

## The robot and the task

{ROBOT}
- **Control.** 11-D whole-body targets at 15 Hz. The full contract is in
  `docs/fridge_robot.py`: read it first.
- **Physics.** The door is a real hinged body (30–50 kg, damped). It opens only if the robot
  pulls it open, usually by grasping the handle and moving along the door's arc. Handles
  differ between models (long vertical bars, short bars, bars on mounts). Fridge types:
  French door (two doors; either counts), side-by-side (the fridge door), bottom freezer
  (the top door). Kitchens can be narrow.
- **Success.** The door's final angle is at least 60° (also reported at 85°; aim for fully
  open). The program decides when it is done: return from `run(robot)`.

## The model: GPT-6 Astra, and nothing else

`robot.astra(...)` (and `astra_start` / `astra_poll` / `astra_wait`) is the only model the
robot has. It sends images and a prompt to GPT-6 Astra through Codex (your own login), with
every tool switched off: it can only look and answer. You choose, per call:
- the prompt, the images (and what you draw on them), and an optional JSON schema for the answer;
- `instructions`, which replace the model's system instructions;
- reasoning `effort` (low, medium, high, xhigh) and service `tier` (default, fast, priority);
- `n`: send that many identical requests at once and take the first answer (1–4);
- `session`: continue one conversation across calls instead of starting fresh.

There is no Gemini, no detector, no segmenter. Your code may use numpy, scipy, OpenCV and
Pillow for geometry and drawing, but not to make the task's decisions (see Rules).

**Time is real.** The world does not wait: while your code computes or waits on Astra, the
simulation runs on, the robot holding its last command. An Astra answer arrives its real
latency after the request, in world time, and the robot can keep moving meanwhile
(`astra_start`, then `act()`, then `astra_poll`). An episode is 90 s of world time at most
and is scored on its time (`robot_time_s`). `./lab profile` measures Astra latency for any
settings, without the robot.

Astra calls count against the weekly Codex usage you share with this session (the cap is
below), so a policy that sends many large requests is expensive to research.

## Scoring

Your final policy is scored by us on a **held-out test set: fridge models and kitchens you
never see**. Ranked by:
1. success rate at 60° and at 85°;
2. time per episode (world time, everything included);
3. Astra usage per episode (calls and tokens).

**Per-episode limits:** {LIMITS}.

## Rules

This is research, and the result is only worth something if it is honest. If something
is ambiguous, write it down in `research_log.md` and make the conservative choice.

1. **Astra decides.** Every decision that depends on the particular fridge and scene comes
   from Astra's answers in that episode: which door, where to grasp it, which way it swings
   and about where, how to approach, whether a grasp or pull is going wrong and what to do
   about it, and when it is done. Your code may turn those answers into motion (inverse
   kinematics, depth to 3-D points, trajectories), measure things for Astra (gripper width,
   depth along a line, tracking error, how far the door moved) and enforce safety limits.
   It may refine an answer locally (snap a chosen pixel to the nearest depth edge within a
   few pixels), but it may not find the handle, the door or the hinge on its own, or choose
   between options Astra did not choose between.
   - We check this: the final policy is also run on the test set with each episode's Astra
     answers swapped for answers recorded on a different fridge. A policy whose code does
     the deciding will still succeed there; one where Astra decides will fail. We report both.
2. **Don't look for the val or test sets.** Don't open or search for the RoboCasa test split
   (styles and layouts 1–10, or its fridge models), the lab's private directory,
   `scenes.json`, or any val or test results beyond what `./lab val` prints. Don't build
   simulator environments yourself. Use `./lab` only.
{WEB}3. **The policy uses the robot API only.** No simulator state, no files from runs, no hidden
   channels. Don't try to get around the sandbox, the budgets or the time accounting.
4. **No lookup tables of specific fridges.** No per-model or per-scene geometry, poses or
   IDs, in code or in Astra's instructions. General priors are fine ("handles are near the
   free edge of a door").
5. **Don't modify the lab** (`./lab`, `docs/`, the code it runs). Reading the lab's source
   in the ChefBot repo is fine.
6. **Report honestly**, including failures, negative results and anything borderline.

## Your tools

`./lab` (run `./lab -h`):
- `./lab scenes`: the dev scenes (fridge doors, their type and model).
- `./lab run skills/X --scene dev05 --seed 0`: one episode. Results go to
  `runs/<job>/<scene>_s<seed>/`: `result.json`, `video.mp4`, `keyframes.jpg`,
  `skill_log.txt`, `images/`, `astra.jsonl` (every request's settings, latency and answer),
  `calls.jsonl`, and `trace.jsonl` (ground truth for *your* debugging only; the policy can
  never see it).
- `./lab suite skills/X --scenes all --seeds 0,1 --no-video`, then `./lab wait <job>`.
- `./lab val skills/X`: success and time on the held-out val scenes, aggregate only.
  **{VAL_SUBMISSIONS} submissions for the whole session.**
- `./lab profile --prompt ... [--image runs/.../x.png] [--effort low] [--tier fast] [--n 2]`:
  Astra latency, no robot.
- `./lab budget`: val submissions left, time left, Codex usage.

Episodes are slow and costly compared with a plain program (each one makes real Astra
calls), so start with a few scenes and seeds, and widen when something works.

## How to work

1. Read `docs/fridge_robot.py` and `skills/example/`. Run the example on a couple of scenes.
2. Measure before you design: what does Astra get right from one look, how long does a call
   take for each setting, and where does an episode's time go.
3. Iterate: pick the biggest failure or the biggest time sink, make one change in a new
   version (`skills/v2`, ...), compare against the previous best on the same scenes and
   seeds, keep it or not based on the numbers.
4. Write each experiment up in `research_log.md` (hypothesis, change, result, kept or not).
5. Keep every version as its own `skills/vN` folder; commit with git if you can.
6. Make it work, then fast, then economical.

**Session.** About {HOURS} hours of wall time. Codex usage is capped at {USAGE_CAP}% of the
weekly limit (shared with the policy's own Astra calls). When less than about 40 minutes are
left, `./lab` responses tell you to wrap up.

**Wrap-up (required).** Put your best policy in `skills/final` and run it once with `./lab`.
Write `REPORT.md`: your approach; what worked and what didn't (with numbers); the dev and val
results of the final policy; its time and Astra usage per episode; what Astra decides in it
and what the code does; its known failure modes; whether it could run on a real robot.
