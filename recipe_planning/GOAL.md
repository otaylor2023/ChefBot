# Recipe planning: the goal

Agreed with Jack on 2026-11-05. Judge every change against this, not against a process.

**Goal.** Turn any recipe from the web into faithful, complete, action-ready instructions that a later step can
translate into whatever small set of skills the robot has, and know how reliably each model (local and hosted)
does it. It is both a tool (its output feeds ChefBot's task planner later) and an evaluation good enough to report.

## What "done" means

1. **Faithful.** Every step, ingredient and dependency traces back to the source text. Gaps are flagged, never
   invented.
2. **Action-ready.** Each step states what is acted on, with what equipment and settings, after which steps, and how
   completion is recognised. Anything the recipe leaves open is an explicit question, not a silent default.
3. **Measured on a real corpus.** Hundreds of recipes, scored against reference answers (gold), not only "passed
   the checks". Two sets, reported separately:
   - **broad**: many sites and cuisines, including baking and dough (parser quality, difficulty)
   - **robot-plausible**: stovetop, eggs, simple assembly (what ChefBot might actually cook)
4. **Models compared.** Per model: success rate, accuracy against gold, cost, latency, and which recipes are hard.
5. **Independent of the robot.** The robot's skill list is a swappable config. Instructions never change when skills
   do. Only the feasibility report does.

## Not the goal (yet)

Executing recipes on the robot or in simulation; compiling to ChefBot task YAML. That comes after (1)-(5).
