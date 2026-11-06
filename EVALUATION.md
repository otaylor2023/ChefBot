# Agent evaluation: sets, conditions, results

Every change to the agent harness on branch `harness-memory-planner` is measured on the sets below, before and after,
in the symbolic kitchen (ground truth from the simulator; the agent never sees it). Each evaluation runs from a git
worktree frozen at the commit it measures, so later edits can't leak into it. Metrics come from each run's
`run.jsonl` (`python -m chefbot.scripts.eval_metrics`).

## Sets

| Set | Jobs | After | Measures |
|---|---|---|---|
| **Cooking** (main) | the 17 cooking jobs (`chefbot/jobs`), 2 seeds | the task set of AGENT_SPEC | success, turns, cost |
| **Requests** | the same 17 from their `prompt:`, plus 3 ambiguous (`req_*_ambiguous`, `req_sausage_where`) with a simulated user, and 1 impossible (`req_pancakes_impossible`) | HW1's persona evaluation; SayPlan's plan verification | planning from a request against the hand-written plan; asking when it should; refusing when it should |
| **Recovery** | 7 jobs with injected failures, disturbances, a blocked path, fragile eggs (`fried_egg_failures`, `_disturbed`, `_blocked`, `mixed_grill`, `scrambled_eggs_fragile`, `breakfast`, `stir_fry_served`), failure noise `on`, 3 seeds | BUMBLE (recovering from failed skills); the Orchestration Gap's recovery category | success; the same failing call repeated (inside and beyond the window); complete refused and looping |
| **Finding** | 4 stale-map jobs (`find_*`): objects moved into closed cabinets or gone before the run, 3 seeds | MessyMem's Finding family | success, finding calls, turns |
| **Memory** | 3 A-then-B sequences in one kitchen (`seq_*`): B run with A's memory and without, 3 seeds | MessyMem's long sequence without a memory reset | B's success, finding calls, turns |
| **Window** | 6 long jobs (`breakfast`, `scrambled_eggs`, `_fragile`, `mixed_grill`, `fried_egg_disturbed`, `find_two_moved`) at windows 4, 8, 16, 32, 2 seeds | the papers' compact-record vs raw-history question | success, turns, tokens |

Metric columns (`eval_metrics`): `succ` ground truth; `turns`; `cost$` LLM (executor + planner); `err` failed calls per
run; `rep<=8` / `rep>8` the same failing call made again within / beyond 8 calls; `refuse` complete refused by the
check; `loop` the longest run of failing complete calls; `find` calls before the first pick_up; `ask` planner questions.

## Steps and results

Symbolic kitchen, `claude-sonnet-5.5` (executor and planner), `claude-haiku-5.5` (judge, simulated user). Each cell:
successes / runs. Commits on branch `harness-memory-planner`; the baseline is `agents-skills-registry` (73a9c15).

### What each step changed

| Step | Commit | Change | Why (the evidence before it) |
|---|---|---|---|
| 0 | 73a9c15 | baseline: executor with hand-written plans | |
| 1 | 505efde | **planner**: a user's request to a checked plan (ask_user, cannot_do, submit_plan); replanning | runs could only start from a hand-written plan |
| 1b | 7ea2c06 | planner prompt: how cooking works in this kitchen; no invented limits; ask where to serve | step 1's 8 refusals of fried eggs ("can't crack an egg by hand") and req_sausage_where never asking |
| 1c | d78805f | planner prompt: the kitchen's cooking times and when food overcooks; no staging steps; always ask when the dish or where it ends up isn't said | step 1b's scrambled eggs (3 of 6 lost: both eggs staged on the counter, no timing) and req_egg_ambiguous not asking |
| 2-3 | 05abeb7 | step record keeps failed attempts; complete blocked after 2 refusals until the kitchen changes; SEARCHING only what was asked for | RoboCasa mixed_grill: 5 complete calls in a row against the check |
| 4-5 | 782ad2e | LEARNED BY DOING (what failed on what, what fixed it, whole run); kitchen memory across runs (`--memory`) | MessyMem: interaction-derived, persistent memory |
| 6 | 0f29f82 | finish refused once while a door the robot opened is open | every remaining finding failure was a cabinet left open |
| 1d | 84b8b3a | planner's ROBOT list keeps each skill's first sentence whole (it was cut at a colon: "align: Fine positioning:", "open: Open a fixture:") | found reading the planner's input for the demo page |

### Results

| Set | Baseline | Step 1 | Step 1b | Steps 2-3 | Steps 4-5 | Step 6 | Step 1c | Step 1d (final) |
|---|---|---|---|---|---|---|---|---|
| Cooking, hand-written plans (17 × 2) | **33/34**, $15.38 | | | | | | | |
| Requests, planned from the prompt (21 × 2) | | 29/42, $17.62 | 32/42, $19.86 | | | | 39/42, $19.30 | **41/42**, $19.83 |
| of which ambiguous, with a simulated user (3 × 2) | | 2/6 | 3/6 | | | | 6/6 | **6/6** |
| of which impossible, refused (1 × 2) | | 2/2 | 2/2 | | | | 2/2 | **2/2** |
| Ambiguous requests without a user (3 × 2) | | 0/6 | | | | | | |
| Recovery, failure noise on (7 × 3) | 20/21, 3.6 err/run | | | 20/21, 3.5 | 21/21, 3.0 | **21/21**, 3.5 | | |
| Finding, stale map (4 × 3) | 8/12, $3.06 | | | 10/12, $2.57 | 11/12, $2.50 | **12/12**, $2.43 | | |
| Memory sequences, B with A's memory (3 × 3) | | | | | 9/9, 18.3 turns, 5.8 finding calls, $0.75 | **9/9, 17.2 turns, 4.9 finding, $0.69** | | |
| Memory sequences, B without memory (3 × 3) | | | | | 7/9, 34.2 turns, 18.8 finding, $1.54 | 9/9, 31.1 turns, 13.6 finding, $1.37 | | |

Step 1d against 1c: 41/42 against 39/42. Every failure in both is a scrambled egg at the 30-minute limit or not
scrambled (1c: 3, in different jobs and seeds; 1d: scrambled_eggs seed 1, 183 turns, the fridge left open), and the
scrambled runs flip both ways between the two, so the 2 runs are within the run-to-run spread, not the fix's doing.
The other 36 runs succeed in both with about the same turns.

Requests at step 1c, the 17 cooking jobs alone: 31/34 planned from the request, against 33/34 from the hand-written
plans (baseline). The 3 failures are scrambled eggs (two at the 30-minute limit, one not scrambled): the planned
scrambles take 120 to 220 turns against about 140 for the hand-written ones.

RoboCasa (MuJoCo physics, video): hand-written plans 3/4 (fried_sausage, stir_fry, fried_sausage_poured; mixed_grill lost
to a complete loop, before step 2); fried_sausage planned from "Can you fry the sausage and put it on the plate?" (step 6
code): success, 54 turns, $0.33, the same five steps as the hand-written plan.

### Window size (steps 4-5 code; 6 long jobs × 2 seeds)

| Window (calls verbatim) | Success | Turns | Cost |
|---|---|---|---|
| 4 | 6/12 | 124 | $7.65 |
| **8** (kept) | 10/12 | 96 | $7.00 |
| 16 | 11/12 | 93 | $9.34 |
| 32 | 11/12 | 92 | $12.77 |

At 4 the agent overcooks, runs out of turns and splatters (it loses track). 8 against 16 differ by one run (a
fried_egg_disturbed at the 150-turn limit); 16 against 32 by none. Both shared find_two_moved failure was a cabinet left
open (fixed in step 6). Across the 4 RoboCasa runs, a failing call was repeated after it left the window once and
inside it 16 times: the window wasn't what failed.

### Failure analysis

- **Planner grounding (step 1):** 5 runs refused at the start and 3 aborted at a replan, all fried eggs: the planner
  believed the robot "can't crack an egg by hand" (the prompt's own words); this kitchen fries a whole egg. Fixed by
  giving the planner the kitchen's cooking rules (step 1b).
- **Not asking where to serve:** req_sausage_where served on the counter both times (step 1).
- **Doors left open:** every remaining finding failure, and the memory sequences' failures without memory (step 6).
- **Timing:** breakfast's egg overcooked (recovery baseline, window 32); mixed_grill's sausage burnt while the agent
  waited on the steak (planner run).
- **Long scrambles:** scrambled_eggs hit the turn limit once (211 turns, step 1) and 3 of 6 scrambled runs failed at
  step 1b: the planner staged both eggs on the counter before cracking (extra carrying) and gave no timing (step 1c).
- **Asking is not reliable:** req_egg_ambiguous asked both times at step 1 and never at step 1b (step 1c's rule).

### Cost

About $170 in total for every evaluation above (about 340 runs), plus $1.78 for the RoboCasa videos.

## Where the runs are

Every evaluation above is in `runs/evals/<step>__eval_<time>_<label>/` (gitignored): `results.json` and one folder per
run (`run.jsonl`, `start.json`, `end.json`, `planner.json`, `report.html`, every LLM request). To re-read one:

    python -m chefbot.scripts.eval_metrics runs/evals/baseline__eval_2026-11-06_020918_noise-task runs/evals/step6__eval_2026-11-06_021845_noise-task_finding --compare

To run a set again on the current code:

    python -m chefbot.scripts.eval_agent --tasks find_mug_moved find_egg_moved find_two_moved find_not_there --runs 3
    python -m chefbot.scripts.eval_agent --plan-from prompt --runs 2          # every job, planned from its request
    python -m chefbot.scripts.eval_sequences --runs 3                         # memory across runs
