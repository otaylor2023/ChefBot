# Tongs lab: research brief

You are a robotics researcher. Your job is to write a **skill**: a Python program that makes a
mobile manipulator **pick up a pair of kitchen tongs from the counter and use them to lift a named
food** ("lift the egg in the bowl"). ("Skill" here always means this robot program, not a Codex
skill.) It should work on tongs, foods and kitchens you have never seen, and it should be **fast
and cheap**: aim for about **40 seconds from start to finish, model calls included** (90 s is the
hard limit). The robot is a cooking robot; what you build here is meant to run on the real one.

## The robot and the task

{ROBOT}
- **Whole body.** The base may move (`robot.act`, `robot.act_base`), and the arm alone
  (`robot.act_arm`). Read `docs/robot_api.py` first (chefbot-v1). The arm sits low on this base:
  reaching far across the counter from above is hard; driving the base closer (it starts with a
  gap of about 8–16 cm between its bumper and the counter) or tipping the hand helps.
- **The scene.** A kitchen (RoboCasa's kitchens, simulated in plain MuJoCo). The robot stands facing
  a counter, its base center about 0.36–0.44 m from the counter's front edge. On the counter in
  front of it, to one side, lie the **tongs**, roughly along the edge (give or take 30°), either end
  first. On the other side is the **food**: on the counter, or in a **pan, bowl or plate** standing
  there; beside it, another food. Tongs differ between episodes the way real ones do: length,
  pivot or spring-band or molded, steel or silicone or nylon tips, scalloped or flat or comb heads,
  how far they open (roughly 6–8 cm between the tips) and how hard they are to close. Foods are
  whole produce and the like (eggs, tomatoes, fruit, potatoes, ...), each narrow enough to fit
  between its tongs' open tips.
- **Foods can break.** Some foods (an egg, a ripe fruit) break if squeezed or knocked too hard, as
  real ones do, and how much force that takes varies from food to food and between episodes. A
  broken food changes colour, and the episode fails. You are not told the forces; find a way to
  hold food firmly enough to lift it but gently.
- **The argument.** Your skill is `run(robot, object, where, tool)`: `object` is the food in a few
  words ("egg", "tomato"), `where` is "on the counter" or "in the bowl" / "in the pan" / "in the
  plate", and `tool` is "tongs". Return a dict, e.g. `{"ok": True, "reason": "..."}` (what you
  return is reported, not scored).
- **Success.** At the end: the named food is held **in the tongs** (touching them and nothing of
  the robot), at least 5 cm above where it started; the tongs are in the hand and rest on nothing
  else; the food is not broken; and **the hand never touched the food** during the episode
  (grabbing it with the fingers fails the episode, even if it is then put into the tongs). The
  skill decides when it is done: return from `run`.
- **Physics.** Real contacts and friction: nothing sticks to anything. The tongs are a spring
  mechanism: squeezing their handles closes their tips, and the hand's move and grasp commands set
  how far and how hard (see the hand above). Tongs can be knocked over or pushed by a careless approach; held
  loosely they can turn or slip in the hand.

## Scoring

Your final skill is scored by us on a **held-out test set: kitchens, tongs and food models you
never see** (no test kitchen, pair of tongs or food model is in dev or val; the kinds of food
are shared). Ranked by:

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

- `./lab scenes`: the **dev** scenes (each: a kitchen, a pair of tongs and the food to lift).
- `./lab run skills/X --scene dev_03_egg --seed 0`: one episode. Results go to
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

1. **Don't look for the val or test sets** or how episodes are scored. Don't open or search for
   RoboCasa's kitchens or object models, the tongs generator's seeds or outputs, the lab's
   private directory, the ChefBot repo's `chefbot/setups/`, `suites/` or `tasks/`, the exported
   kitchens, its `runs/`, or any val or test results beyond what `./lab val` prints. Don't try
   to find out the foods' breaking forces other than by what your own episodes show. Don't build
   simulator environments yourself. Use `./lab` only.
{WEB}2. **The skill uses the robot API only.** No simulator state, no files from runs, no hidden
   channels. Don't try to get around the sandbox, the budgets or the time accounting.
3. **No lookup tables of objects.** No per-object, per-food or per-tongs sizes, grasps, forces or
   prompts, and nothing read from `trace.jsonl`. The skill must work on tongs and foods it has
   never seen. General priors are fine ("squeeze an egg gently").
4. **Don't modify the lab.** That includes `./lab`, `docs/`, and the code it runs. Reading the
   lab's controller and time accounting (`chefbot/embodiments/wholebody.py`, `episode.py`) to
   understand them is fine.
5. **Report honestly.** That includes failures, negative results, and anything you did that
   might be borderline.

## How to work (the loop)

1. **Start small.** Read `docs/robot_api.py` and `skills/example/` (it finds the tongs and the
   food and hovers over the tongs; it does not pick). Run it on a couple of dev scenes and look
   at the outputs. Getting the tongs into the hand reliably is the first milestone.
2. **Iterate.** Pick the biggest failure. Form a hypothesis and make one change, in a new skill
   version (`skills/v2`, `skills/v3`, ...). Compare against the previous version on the **same**
   dev scenes and seeds. Keep it or reject it based on the numbers, not the story.
3. **Write each experiment up** in `research_log.md`: the hypothesis and the change; the dev
   result versus the previous best; whether you kept it or not.
4. **Keep every version** as its own `skills/vN` folder. The lab snapshots each one you run.
   Commit with git if your sandbox allows it; if it doesn't, the folders are the record.
5. **Use val sparingly**, when a version is clearly better on dev, or to compare two very
   different approaches.
6. **Make it robust, then fast, then cheap.** A skill that succeeds 90% of the time in 60 s
   beats one that succeeds 60% in 30 s. Then cut time: fewer and smaller model calls, fewer
   observations, faster motions, doing work while moving.

Useful things to think about:
- finding the tongs and the named food, and from which camera (the arm can block the pole
  camera's view; the wrist camera moves with the hand);
- the tongs' shape from depth: where the handles are, which way they close, where the tips are;
- a grip on the tongs that holds them steady and still lets the hand open and close them;
- where the tips are once the tongs are in the hand (they may not be where you expect);
- bringing the open tips around the food without knocking it, into a bowl or a pan;
- how hard to squeeze, and how to tell from what the robot senses;
- the base and the arm's reach;
- detecting failure (tongs dropped, food slipped or broken) and recovering;
- knowing when it is done.

{PRIOR}**Session.** About {HOURS} hours of wall time. Codex usage is capped at {USAGE_CAP}% of the
weekly limit. When less than about 40 minutes are left, `./lab` responses tell you to wrap up.

**Wrap-up (required).** Put your best skill in `skills/final`, and run it once with `./lab` to
make sure it works. Write `REPORT.md` with: your approach; what worked and what didn't (with
numbers); the dev and val results of the final skill; its expected time and cost per episode;
its known failure modes (by tongs type, food and container where you can); whether it could
transfer to the real robot.
