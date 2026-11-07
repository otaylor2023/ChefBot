# Recipe agent: midway findings and next steps (2026-11-06)

What the recipe agent and RecipeBench (`bench/`) have shown so far, what is weak, and what comes next. Goal and
"done" criteria: [GOAL.md](GOAL.md). How the benchmark works: [bench/README.md](bench/README.md).

## Setup in one paragraph

An LLM extracts a recipe into a strict schema (25-verb closed set, every element carries a verbatim evidence
quote and a provenance). A deterministic verifier (no LLM) checks quote grounding, runs a symbolic state engine,
rejects unstated heat/temperature/time (`unstated_param`) and builds the step dependency graph. If the extraction
has errors, the model gets one guarded repair with an error-code hint; the repair is kept only if the errors drop.
The output (`actions.json`) is robot-independent; feasibility against a skills file (`skills/chefbot_sim.yaml`) is
a separate report.

RecipeBench, frozen as `recipe_eval_v1` (33 recipes: 3 gold, 15 robot-plausible, 15 broad; sha256 per file):
each instance is a recipe x a perturbation (base, reorder, no_headings, noise, injection, no_heat_level, units,
planted_gap), 178 in the full split, 75 in lite, 14 in mini (local GPU). Resolved = no extraction errors AND the
instance's own test (bench/checks.py). Invariance variants are judged against the same model's own base run.
Reference answers: gold by hand for 3 recipes; silver from gpt-6.1-sol for the other 30.

## Results (lite split, prompt 0.7, one run per instance)

| configuration | n | resolved | 1st-try valid | recovered / mistakes | injection | planted gap | cost |
|---|---|---|---|---|---|---|---|
| gpt-6.1-sol, repair + hints | 72 | **97.2%** | 65.3% | 25 / 25 | 100% | 100% | $5.77 |
| gpt-6-luna, repair + hints | 75 | **85.3%** | 56.0% | 30 / 33 | 72.7% | 57.1% | $0.27 |
| gpt-6-luna, repair, no hints | 75 | **85.3%** | 58.7% | 28 / 31 | 81.8% | 71.4% | $0.27 |
| gemini-3.1-pro-preview | 52 | **82.7%** | 38.5% | 30 / 32 | 75.0% | 50.0% | $7.69 |
| claude-sonnet-5.5 | 62 | **80.6%** | 45.2% | 29 / 34 | 66.7% | 20.0% | $6.32 |
| claude-haiku-5.5 | 75 | **80.0%** | 34.7% | 38 / 49 | 90.9% | 42.9% | $0.80 |
| gpt-6-luna, no repair | 75 | **50.7%** | 58.7% | 0 / 31 | 54.5% | 42.9% | $0.18 |

Local, mini split: qwen3.5 9B (thinking off) 50.0% with repair (n=10), 33.3% without (n=9). Gemini and Sonnet
stopped at their spend caps (52 and 62 of 75). sol wrote the silver answers, so its score is an upper bound.
Ingestion (tier 0): 96.9% of 319 saved pages give the full ingredient and instruction text. No instance is left
unsolved by every configuration. Full table: `recipe_eval_v1/bench_runs/leaderboard.md` on batcomputer.

## Update, same day: checker v0.8 (prompt unchanged, 0.7)

Two checker fixes, then hosted reruns on the same 75 lite tests (`recipe_eval_v1/bench_runs_v08/`):
- `unsupported_cooking_step`: a stated cook/bake step must quote a sentence with a cooking word, doneness cue or
  time. Offline it flags all 9 planted-gap cheats in the midway outputs (tomato-egg 5, tamagoyaki 4) and 8
  weakly quoted cook steps elsewhere, which the repair fixes by quoting the right sentence.
- Worded durations ("a minute", "two to three minutes", "half an hour"; "a few minutes" stays vague). This removed
  16 false `unstated_param` errors.

| configuration (checker v0.8) | n | resolved | 1st-try valid | recovered / mistakes | injection | planted gap | cost |
|---|---|---|---|---|---|---|---|
| claude-sonnet-5.5 | 64 | **85.9%** (was 80.6) | 54.7% | 23 / 29 | 70.0% | **60.0%** (was 20) | $5.93 |
| gpt-6-luna, repair + hints | 75 | **85.3%** (same) | 72.0% (was 56) | 19 / 21 | 72.7% | **85.7%** (was 57) | $0.25 |
| claude-haiku-5.5 | 75 | **74.7%** (was 80.0) | 36.0% | 37 / 48 | 72.7% | 28.6% (was 42.9) | $0.76 |
| gpt-6-luna, no repair | 75 | **60.0%** (was 50.7) | 72.0% | 0 / 21 | 63.6% | 42.9% | $0.18 |

**Correction to the midway headline.** Part of the 51% → 85% repair effect was the repair undoing the checker's own
duration false positives. With the checker fixed, luna without repair scores 60.0%, so one repair is worth
**+25 points (60 → 85)**, not +34. Rescoring the saved midway outputs with the new checker (no new model calls,
`bench_runs_rescored_v08/`) gives the same picture: luna no-repair 61.3%.

**Run-to-run noise is about ±5 points.** Haiku's drop is not the new check: none of its 19 changed verdicts
involve `unsupported_cooking_step`. 9 tests flipped to pass and 10 to fail, almost all from flaky JSON
(`schema_error`, then a retry) or a different step granularity. Single-run differences between models below ~5
points should not be read as real; pass^k (next step 4) is now the priority for comparing models. Sonnet ran 64 of
75 tests here vs 62 before (spend cap), so its row compares slightly different subsets.

### pass^3 (same 75 tests, three runs each, checker v0.8)

| configuration | mean single run | pass^3 (all 3 runs) | pass@3 (any run) | flaky tests | never solved | first-try schema failures |
|---|---|---|---|---|---|---|
| gpt-6-luna, repair + hints | 89.8% | **78.7%** | 100% | 16 | 0 | 0 of 225 runs |
| claude-haiku-5.5, repair + hints | 79.2% | **59.7%** | 95.8% | 26 | 3 | 134 of 222 runs |

(Invariance variants judged against the same run's own base, as in the leaderboard. Haiku: 72 of 75 tests have all
three runs; the spend cap stopped the rest.)

- **One run overstates reliability.** A model that resolves 85-90% of tests in a single run solves only 79% (luna)
  and 60% (Haiku) every time. No luna test fails all three runs: every failure is intermittent, so a
  retry-until-the-checks-pass policy would recover them (the checks already know when an output failed).
- **Haiku's flakiness was mostly our harness.** Its first answer was cut off at the 8,000-token cap in about half
  of all runs (median output exactly 8,000; luna's is ~3,900) because OpenRouter counts Haiku's hidden reasoning
  (3-5k tokens) toward `max_tokens`. The truncation retry usually rescued it, but at the cost of a failed first try
  and, sometimes, the run. Fixed: OpenRouter calls now get reasoning headroom on top of the answer cap
  (`OPENROUTER_REASONING_HEADROOM`, default 8,000; billed only if used).

**Re-measured with the fix** (Haiku, same 75 tests x 3 runs, `recipe_eval_v1/bench_runs_v08_headroom/`, $1.74):

| claude-haiku-5.5 | mean single run | pass^3 | pass@3 | flaky | never solved | first tries cut off / schema-invalid |
|---|---|---|---|---|---|---|
| before (8k cap incl. reasoning) | 79.2% | 59.7% | 95.8% | 26 | 3 | 134 of 222 |
| after (reasoning headroom) | **93.8%** | **85.3%** | **100%** | 11 | 0 | 14 of 225 |

A one-line harness fix moved Haiku from last to first on reliability (luna: 89.8% single run, 78.7% pass^3). The
earlier model ranking measured our token budget, not the models. Lesson: before comparing models, check that the
harness gives every model the same chance (budgets, output modes, retries); per-model failure codes
(`first_errors`) found this in minutes. Planted gap is now Haiku's weakest variant (3 of 7 always solved).

## Findings

1. **The verifier + one repair is the main effect.** Same model (luna), fixed checker: 60.0% without repair, 85.3%
   with (+25 points). Midway numbers with the old checker: 50.7% → 85.3%. Only 35-72% of first tries are valid; one
   repair recovers 78-100% of the mistakes.
2. **Hints did not matter.** luna scores 85.3% with and without the error-code hints. The gain comes from a precise
   failure signal and a second attempt, not from better repair instructions.
3. **Models fill gaps instead of reporting them.** The planted-gap variant deletes the step that cooks an
   ingredient. The prompt says "Represent the recipe AS WRITTEN. Do not fix it", yet 4 of 5 hosted models put a
   cooking step back on the tomato-egg stir-fry (luna marks "immediately add the eggs" as a stated `cook` step).
   Planted gap is the weakest variant (Sonnet 20%, Haiku 43%, Gemini 50%, luna 57%). A deterministic check that a
   cooking step's sentence actually says "cook" turns the repair loop against this: planted gap rises to 86% (luna)
   and 60% (Sonnet) with the same prompt. The tamagoyaki case stays hard: models cite an unrelated tip ("the egg is
   getting cooked too fast") as evidence, which passes a word check.
4. **A cheap model plus the harness gets close.** luna: 85% for $0.27 per 75 tests; sol: 97% for $5.77.
5. **Prompt injection is mostly but not fully resisted** (67-100% across hosted models).
6. **Recipes leave a lot open.** sol's outputs on the 12 lite base recipes carry 68 open questions (unstated heat,
   "a few minutes", unallocated salt); 11 of 12 recipes have at least one.
7. **The robot can do part of a typical recipe.** Tomato-egg stir-fry against `chefbot_sim`: 7 of 17 steps
   supported, 5 partial (egg-cracker or whisk gadget; stovetop cooking whose doneness comes from simulator state,
   not vision), 5 unsupported (no knife, no shaker, no lid).

## Recipe chooser and PantryBench (next step 8, v0)

`recipe_ir/choose.py` + `choose_recipe.py`: pick a recipe from the pantry, constraints and the robot's skills file.
Deterministic, no LLM. Hard constraints (diet, exclusions, max missing items, `--robot strict`) are checks: a recipe
that breaks one is never returned. Missing items are reported ("ask, don't buy"), never ordered.

`bench/pantry.py` builds 480 scenarios from 80 real corpus recipes: the target's ingredients typed three ways (exact,
terse head noun "onion", synonym "green onions"), 0 or 1 removed, 15 distractor items.

| pantry names | recall@3 (no item removed / 1 removed) | hid a missing item | asked for more |
|---|---|---|---|
| exact | 100% / 100% | 0% / 0% | 2.5% / 1.2% |
| synonym | 100% / 100% | 0% / 0% | 2.5% / 2.5% |
| terse | 91% / 81% | 0% / 5.0% | 51% / 45% |

Diet constraints over the whole corpus: 0 violations (vegetarian 46 returned, vegan 20). Terse names are the hard
case: the matcher is conservative ("cheese" is not assumed to be cream cheese) but "pepper" can still cover
"jalapeño pepper". An LLM matcher for ambiguous names is the next step there.

## Compile to a ChefBot job (next step 7)

`recipe_ir/to_job.py` + `to_job.py`: `actions.json` → `chefbot/jobs/*.yaml` format (goal, steps with goal,
done_when, depends_on, hint). The recipe's words stay the goal; done_when comes from the completion cue; the skills
report goes into hints and `source.needs_a_person`. A test loads the result with ChefBot's own `Plan` model.
Example: `examples/jobs/tomato_egg_stir_fry_from_web.yaml`. A job still needs a kitchen-specific `symbolic:` block
(objects, success) before it runs in simulation.

## Known issues

- Fixed in v0.8: `unstated_param` false positive on durations written in words ("Beat eggs for a minute").
- Silver answers are biased toward sol; only 3 gold recipes. Base-split scores against silver are agreement, not
  accuracy.
- One run per instance, so no pass^k / variance yet.
- Spend caps overshoot with parallel workers (Sonnet $6.32 on a $5 cap, Gemini $7.69 on $6).
- Local runs are limited by GPU heat (RX 6800 XT reached 103-106 C junction): mini split, duty cycle and
  `--max-gpu-temp` guard only; never unattended.
- Some sites block scraping (budgetbytes.com behind Cloudflare).

## Next steps

1. ~~Faithfulness check for cooking verbs~~ done (v0.8). Next: catch evidence taken from tips rather than steps.
2. ~~Fix duration parsing~~ done (v0.8).
3. **Grow the gold set** (target 15+, both splits) so the main results are against human answers.
4. ~~pass^k~~ done for luna and Haiku, and Haiku re-measured after the headroom fix (above).
5. **Full split (178)** for luna and one Claude model with an exact budget check before each call.
6. **Local models** after setting a GPU power limit; then gpt-oss 20B and gemma on mini.
7. ~~Compile `actions.json` to ChefBot jobs~~ v0 done. Next: write the `symbolic:` block for one web recipe in
   the sim kitchen and run it end to end.
8. **Recipe chooser:** v0 done (deterministic, PantryBench). Next: an LLM matcher for terse/ambiguous names and
   fuzzy preferences, graded by the same PantryBench; then chooser → extraction → job in one command.

## Reproduce

On batcomputer: `cd .../ChefBot_recipe/recipe_planning && bash bench/report_run.sh` (under an hour: hosted lite in
parallel with spend caps, local mini with the GPU guard, then ingest and leaderboard). Data and results live in
`Engineering_AI_Agents/recipe_eval_v1/` (not in git).
