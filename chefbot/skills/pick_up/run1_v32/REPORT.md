# Pick skill research report

The final skill is **v9**, copied unchanged to **`skills/final/main.py`**. It passed **18/18 held-out validation episodes in each of two runs**. These repeat the same 18 cases; they are not 36 independent cases. Its recorded dev result is **115/120 (95.83%)**, averaging **18.22 seconds**.

V32 was the strongest development candidate: 180/180 successes, 16.99 seconds mean, and lower model cost. However, it scored 17/18 on validation twice. Because unseen-object success ranks first, the final selection follows the repeated validation evidence. This small validation set does not prove v9 is universally better; the tradeoff and all results are preserved rather than claiming that the development gains generalized.

## Final approach

`run(robot, object)` uses only the documented robot API and `act_arm`. Grounding DINO proposes boxes for the named object. When detection is missing, low-confidence, or ambiguous, a Gemini Robotics-ER call supplies a semantic point. SAM 2 segments that point; otherwise it segments the strongest detector box. Invalid semantic output falls back to the detector.

The supplied pole-camera intrinsics and pose transform depth into arm-root coordinates. Mask erosion and robust point-cloud bounds reduce boundary contamination. A local height histogram estimates the support surface. Horizontal PCA supplies a narrow closing direction, and the equivalent finger orientation is chosen toward the reachable wrist half-turn. The grasp height is near the object's middle, with a small minimum support clearance. No category-specific sizes, grasps, or prompts are used.

The arm approaches above the object, descends, closes, and requests a 13 cm lift. Measured pose errors control bounded settling, including a plateau stop when residual errors are moderate. One empty-gripper retry returns to the initial pose, observes again, and switches to an orthogonal direction when measured geometry permits. All motion is arm-only. The final policy does not include v32's circle-center correction, tilted reach recovery, or shorter waits.

## Development and negative results

There are 30 dev scenes. Final v9 evaluation uses seeds 0–3: 120 episodes from `dev-140608-c03b` and `dev-141705-a029`. Later v32 evaluation uses seeds 0,2,4,7,8,9; seeds 8/9 were fresh placements/calibration during this continuation, but reused development object categories.

| Experiment | Success | Mean world time | Finding |
|---|---:|---:|---|
| Example, two episodes | 0/2 | 47.87 s | Hover only; includes cold detector startup |
| v1 → v3, initial four | 0/4 → 4/4 | 30.95 → 35.8 s | Reachable wrist symmetry and convergence enabled picking |
| v3 → v4, same 60 | 51/60 → 51/60 | 42.26 → 15.64 s | Practical tolerances removed unnecessary waits |
| v6 recovery subset vs v4 | 11/12 vs 8/12 | 24.69 vs 21.23 s | One retry recovered three failures |
| v8 identity subset vs v4 | 9/10 vs 5/10 | 21.03 s for v8 | Semantic identity resolved detector mistakes |
| **v9/final, seeds 0–3** | **115/120** | **18.22 s** | Median 13.67 s; p95 46.80 s; 14 episodes >30 s; maximum 90 s |
| v25 vs v28, same 240 | 238/240 each | 18.41 vs 18.13 s | Flash-Lite matched Robotics-ER on this dev comparison at 7.13× lower cost |
| v28 → v29, matched 120 | 118/120 → 119/120 | 18.91 → 16.74 s | Geometric recovery fixed two failures but introduced a can slip |
| v29 → v31, same 120 | 119/120 → 120/120 | 16.74 → 16.01 s | Shorter waits recovered that slip without measured losses |
| v30 vs v31, closure-only 30 | 30/30 each | 19.89 vs 19.44 s | Fixed closure dwell saved modest time |
| v31 → v32, same 180 | 179/180 → 180/180 | 18.02 → 16.99 s | Stalled-lift stopping reduced episodes >30 s from 21 to six |
| v32, fresh seeds 8/9 only | 60/60 | 19.58 s | One long recovery took 60.73 s |

V32 combines always-on cheap semantic identity, whole-object box selection, broad/unreachable-mask checks, constrained cylindrical center fitting, reach recovery, and shorter waits. Its 180/180 dev result is promising, but the repeated 17/18 validation outcome is a negative generalization result relative to v9. It remains under `skills/v32` for future research, not in the final submission. No validation failure identities or details were inspected, so the cause of this regression is unknown.

Other rejected or limited experiments:

- Measured-pose lift compensation could move targets outside reach: 57/60 versus its parent's 59/60.
- A silhouette-axis rule traded failures: 114/120 versus 115/120.
- Tilted pregrasp recovery scored 115/120 versus 117/120; lowering an unreachable hover scored 18/20 versus 19/20.
- Two-camera semantic selection added time/cost without improving its tested success set.
- Small semantic output budgets truncated responses, and point-only SAM could select a label. Later versions addressed these issues, but those changes were not independently validated as a replacement for final v9.
- Nearly open post-close width preceded both a dropped can and a successful lift. A proposed width-only rejection was not implemented.

Small differences can reflect model outputs, latency, and contact variability. Interrupted runs are not treated as completed comparisons. All source versions, hypotheses, job IDs, failures, and reporting corrections remain in `research_log.md` and `skills/vN`.

## Held-out validation and selection

Only permitted aggregate output from `./lab val`/status was inspected. All five submissions were used.

| Version / run | Success | Mean world time | Mean model cost |
|---|---:|---:|---:|
| v4 | 14/18 | 18.03 s | $0 |
| **v9, original** | **18/18** | **16.36 s** | **$0.0015** |
| v32, original final candidate | 17/18 | 16.87 s | $0.0003 |
| **v9, unchanged repeat** | **18/18** | **16.15 s** | **$0.0014** |
| v32, unchanged repeat | 17/18 | 16.57 s | $0.0003 |

Costs are rounded as printed by the aggregate interface. Each validation run had zero harness errors and zero sandbox blocks. After the first v32 regression, the comparison rule was recorded before the repeats: favor total validation successes; if tied, favor v32's development robustness and lower cost. Neither candidate was modified. The result selected v9. No hidden failure information was used to tune parameters. Repeated validation scores are correlated and should not be interpreted as an expanded independent test set.

## Expected time, cost, and remaining failures

Final v9 averages 18.22 seconds and $0.000931 per episode on its 120 dev episodes. Its two validation runs average about 16.26 seconds and approximately $0.0014–$0.0015 per episode. Model calls are conditional, so easy detections incur no LLM cost. Local detection and segmentation have no metered dollar cost but consume time. These times include computation and model latency in the continuously running world.

The approximately 15-second target is approached, but the requested 30-second maximum is **not achieved**. Fourteen of 120 dev episodes exceeded 30 seconds; one reached the 90-second limit. Final v9's known dev failures include marginal lift clearance and unreachable top-down grasps, particularly at higher supports. It has no tilted reach recovery. A longer or cheaper policy was not selected solely for stronger dev numbers when repeated validation favored v9.

Other risks include partial-depth center bias, wrong-object detections, semantic coordinate errors/truncation, SAM selecting an object part, tipping, rolling, and slipping. Gripper width does not establish target identity or fingertip-pad contact. The returned `ok` is only a width proxy; scored results, not this return value, are used here. No independent visual verification follows the lift.

The architecture could transfer to hardware because it uses calibrated RGB-D and ordinary proprioception. It is not ready for unattended deployment. Transfer requires measured finger geometry, handling noisy/missing/reflective depth, collision and force limits, and testing real friction, timing, and controller behavior.

## Final artifact, integrity, and reproduction

Entry point: `skills/final/main.py::run(robot, object)`. Dependencies are NumPy, SciPy, OpenCV, and Python JSON. No assets or external files are required. Final source is byte-for-byte v9, lab digest `3c8cfd9643f2`; main.py SHA-256 is `e74997e7f1ac5a912e44de069309ccff91767695db82885f58cdfc6d41a78619`.

```sh
./lab run skills/final --scene dev_11_can --seed 7
./lab suite skills/final --scenes all --seeds 0,1,2,3 --no-video
```

The required final-v9 smoke test **passed**: `dev-232205-f869`, can seed 7, held with **7.9 cm lift**, **58.07 s**, **$0.00514**, and **zero sandbox blocks**. Its keyframes were inspected. This additional case is outside the 120-episode benchmark; including it gives **116/121 dev successes**, averaging **18.55 s**. The earlier v32 smoke test also passed, but is not counted as a final-v9 test. Final metrics and provenance are saved in `analysis/final_metrics.json`; the v31/v32 comparison remains in `analysis/v31_v32_comparison.json`.

No simulator state, hidden channels, base commands, object/category lookup tables, private sets, object models, test data, or sandbox workarounds were used. Development images, skill logs, and results were inspected. Earlier sessions read initial dev trace rows as recorded in the research log; no trace-derived constants were incorporated. This continuation read no traces or validation artifacts. An accidental early v2 repeat was retained and logged. Version folders and lab snapshots preserve the source history; Git metadata is read-only under the supplied permissions.

Session end budget: $2.4739 of $10 total model spend across all conversations, no validation submissions left, and 1.0% reported weekly Codex usage. No jobs remain active.

Commit status: attempted on explicit request, but Git could not create `.git/index.lock` on the read-only filesystem. The elevated-access retry failed with `approval request failed`. No commit was created; all source versions, the final skill, report, and research records remain in the workspace. The unrelated AGENTS.md modification was excluded from the staging command.
