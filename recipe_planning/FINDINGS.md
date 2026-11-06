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

## Findings

1. **The verifier + one repair is the main effect.** Same model (luna): 50.7% without repair, 85.3% with. Only
   35-65% of first tries are valid; one repair recovers 78-100% of the mistakes.
2. **Hints did not matter.** luna scores 85.3% with and without the error-code hints. The gain comes from a precise
   failure signal and a second attempt, not from better repair instructions.
3. **Models fill gaps instead of reporting them.** The planted-gap variant deletes the step that cooks an
   ingredient. The prompt says "Represent the recipe AS WRITTEN. Do not fix it", yet 4 of 5 hosted models put a
   cooking step back on the tomato-egg stir-fry (luna marks "immediately add the eggs" as a stated `cook` step).
   Planted gap is the weakest variant (Sonnet 20%, Haiku 43%, Gemini 50%, luna 57%).
4. **A cheap model plus the harness gets close.** luna: 85% for $0.27 per 75 tests; sol: 97% for $5.77.
5. **Prompt injection is mostly but not fully resisted** (67-100% across hosted models).
6. **Recipes leave a lot open.** sol's outputs on the 12 lite base recipes carry 68 open questions (unstated heat,
   "a few minutes", unallocated salt); 11 of 12 recipes have at least one.
7. **The robot can do part of a typical recipe.** Tomato-egg stir-fry against `chefbot_sim`: 7 of 17 steps
   supported, 5 partial (egg-cracker or whisk gadget; stovetop cooking whose doneness comes from simulator state,
   not vision), 5 unsupported (no knife, no shaker, no lid).

## Known issues

- `unstated_param` false positive on durations written in words: "Beat eggs for a minute" (duration 60 s) is
  flagged as not in the source. The repair then deletes a correct value.
- Silver answers are biased toward sol; only 3 gold recipes. Base-split scores against silver are agreement, not
  accuracy.
- One run per instance, so no pass^k / variance yet.
- Spend caps overshoot with parallel workers (Sonnet $6.32 on a $5 cap, Gemini $7.69 on $6).
- Local runs are limited by GPU heat (RX 6800 XT reached 103-106 C junction): mini split, duty cycle and
  `--max-gpu-temp` guard only; never unattended.
- Some sites block scraping (budgetbytes.com behind Cloudflare).

## Next steps

1. **Faithfulness check for cooking verbs:** a `cook`/`bake` step must be supported by its quote (adding an
   ingredient is not cooking it). Targets the planted-gap failure directly.
2. **Fix duration parsing** for worded numbers ("a minute", "a few minutes" as unresolved) and re-score with
   `bench.rescore`.
3. **Grow the gold set** (target 15+, both splits) so the main results are against human answers.
4. **pass^k:** 3 runs per instance on lite for luna and Haiku (cheap) to measure consistency.
5. **Full split (178)** for luna and one Claude model with an exact budget check before each call.
6. **Local models** after setting a GPU power limit; then gpt-oss 20B and gemma on mini.
7. **Compile `actions.json` to ChefBot task files** for the fully supported steps, starting with eggs.
8. **Recipe selection agent** (proactive): pick a recipe from user preferences, the pantry and the robot's
   skills, with its own checks (cook from the recipe, spend cap, ask before buying).

## Reproduce

On batcomputer: `cd .../ChefBot_recipe/recipe_planning && bash bench/report_run.sh` (under an hour: hosted lite in
parallel with spend caps, local mini with the GPU guard, then ingest and leaderboard). Data and results live in
`Engineering_AI_Agents/recipe_eval_v1/` (not in git).
