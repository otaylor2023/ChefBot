# Tongs lab final report

## Artifact and outcome

`skills/final/main.py` is **v59**, byte-identical to `skills/v59/main.py` (lab digest `8561927cbe4e`). It implements `run(robot, object, where, tool)` using only the documented robot API. Numbered experimental versions are preserved.

The final skill scores **7/48 development successes (14.6%)**, with **zero broken foods, zero hand–food contacts and zero sandbox blocks** in that primary set. Mean world time is **36.40 s**, including model calls, and mean model cost is **$0.00334**. Successful episodes average **38.10 s**, ranging from **32.80 to 48.67 s**. Median time over all episodes is 38.20 s; early failures lower the overall mean.

Public validation scored **3/24 (12.5%)** initially and **2/24 (8.3%)** in the unchanged final-artifact replication. Pooled across these repeated submissions: **5/48 episode evaluations (10.4%)**, **35.09 s**, approximately **$0.0031**. This is not evidence from 48 distinct held-out scenes. The requested **20–30 s speed target and reliable unseen-tool performance were not achieved**. This is a research artifact, not a deployable cooking controller.

## Approach and prior reuse

The program retains the supplied prior's DINO/SAM perception, calibrated RGB-D geometry, support-plane and tool-axis estimation, food surface fitting, geometric pickup, horizontal use posture and open-paddle registration. It replaces obsolete fractional gripper closure with explicit Franka grasp-force commands.

1. Detect and segment the tool and requested food. A geometry-only direction estimate is accepted when endpoint widths differ by more than 20 mm and a factor of two. With exactly one named-food detection, the model call is omitted; otherwise the original Robotics-ER landmark call resolves ambiguity. Twelve of 48 primary final episodes use no paid model call.
2. Pick up the tongs securely at 20 N. Reduce force while observing handle width to estimate an empty-tool force/width curve and obtain a usable opening. No food/tool-specific force or size table is used.
3. Register the open tips from depth, transfer above the food and descend. Increasing force stops when the measured width exceeds the empty reference by 2 mm; add a 2 N holding margin, capped at 20 N, then lift.
4. Use RGB-D to look for the same food still near its original support. A retry requires a color-consistent candidate within 40 mm laterally and between 20 mm below and 30 mm above its initial height, nonzero gripper width, sufficient time, newly registered open heads and a geometric hand–food separation check.
5. Retry once at the fitted food equator with a 4 N holding margin, capped at 20 N, and lift from the measured flange pose.

The retry is an observed-failure response, not a score query. It triggered 15 complete retries in the primary 48 episodes; three succeeded: pan orange11 seed0, apricot03 seed1 and plate peach17 seed1. The program still reports possession as unverified: camera evidence and hand width do not establish all task conditions.

## Comparative results

These are completed public development results, with infrastructure errors excluded and retained separately. Primary results are fixed; replications never replace them.

| Version | Dev success | Mean world time | Mean model cost | Main finding |
|---|---:|---:|---:|---|
| v18 primary current-hand reference | 6/48 | 34.56 s | $0.00404 | Earlier best reference |
| v18 unchanged replication | 5/48 | 34.56 s | $0.00389 | Lost the primary peach success; one break |
| v26 downward-clearance branch | 6/48 | 37.75 s | $0.00397 | No broad success gain |
| v40 combined lift/turn | 5/48 | 33.65 s | $0.00406 | One break; favorable small sample did not generalize |
| v51 geometry-gated perception | 5/48 | 33.15 s | $0.00305 | Faster and cheaper, limited reliability |
| **v59 final, with visual retry** | **7/48** | **36.40 s** | **$0.00334** | Net two successes over v51 |
| v63 guarded retry candidate | 5/48 | 35.94 s | $0.00296 | One break; not selected |

Relative to v51, v59 gains apricot03 seed1, orange11 seed0 and peach17 seed1, but loses egg15 seed1. The original v18 primary result remains 6/48; its two predeclared full batches total 11/96. This variation, and a failed unchanged-v18 apricot smoke, show that small geometry/timing changes can alter contact outcomes. The observed rates are imprecise, not evidence of robust deployment reliability.

| Public validation submission | Success | Mean world time | Mean cost |
|---|---:|---:|---:|
| v18 | 8.3% | 34.84 s | $0.0043 |
| v26 | 8.3% | 36.74 s | $0.0042 |
| v51 | 8.3% | 33.59 s | $0.0029 |
| v59 initial | 12.5% | 35.29 s | $0.0031 |
| final v59 replication | 8.3% | 34.90 s | $0.0031 |

Only public aggregates were accessed. Successful held-out case identities are unknown. Both validation results are retained. The small observed gain over the earlier reference is uncertain; the replication does not establish robust superiority.

## Negative results and selection limits

The current-hand v9 reference scored 3/24 at 40.21 s. Adaptive calibration and faster descent reduced that to v18's same 3/24 at 34.25 s. Further speed changes often altered contact dynamics and lost successes: direct initial-pose survey scored 0/4 because of arm occlusion; calibration during base travel scored 2/8 versus3/8; taller lifting scored 1/11 versus3/11. Settled force probes plus root-frame motion retained 3/8 but did not produce a useful speed gain.

Direct closed-tool transfer scored 0/12 with one hand–food contact. Lower contact height, stronger volume-based force margins, semantic/proximal grasp selection and broader head registration did not yield a supported general improvement. Cheap food-box selection reduced cost to $0.00091 but scored 1/12 versus3/12; inspected lost controls selected the correct food, so those failures cannot simply be blamed on classification.

A wider retry-identity gate v60 gained a lemon (5/12 versus4/12) but introduced hand–food contact. One registered offset implied a finger-pad target below the counter. A support-plane guard rejected such targets, but related trials still had pan-contact failures above the plane. v63 added only this guard to the conservative v59 policy and scored 5/48 with one break. It was not selected on the observed results; that does not prove the guard caused all regressions.

**The selected v59 has no explicit counter-plane rejection for retry targets.** Its 48 primary dev trials had no hand–food contact, but related variants expose a real unresolved clearance risk. This limitation is not concealed by its better observed success rate. All hypotheses, outcomes and corrections are in `research_log.md`.

## Failure modes and real-robot transfer

| Tool mechanism | Final dev success |
|---|---:|
| Pivot | 7/32 |
| Molded flexure | 0/8 |
| Scissor | 0/4 |
| Spring band | 0/4 |

| Placement | Final dev success |
|---|---:|
| Counter | 5/16 |
| Pan | 1/12 |
| Plate | 1/8 |
| Bowl | 0/12 |

Foods: eggs 3/4, apricots 2/8, orange 1/2 and peaches 1/4; all other tested kinds have zero successes. These descriptive groups mix kitchens, tools and foods and are not causal material comparisons or policy inputs.

Soft mechanisms often lose retention during opening; scissors remain difficult to grasp and actuate. Larger foods can make contact then slip. Containers introduce reach, rim and registration problems. Only 25/48 primary final episodes end with the reported tool-held condition. Width resistance confounds spring/friction state, tool motion and food contact. The visual retry can reject real food after displacement or occlusion, and can fail to register a shifted tool. It does not reliably verify possession or damage.

The API and sensing architecture could transfer to the stated robot for supervised experiments, but the controller is **not ready for unsupervised real-kitchen use**. Tool retention, collision-aware approaches, force adaptation and reliable possession verification remain unresolved. The supplied prior's historical 14/48 dev and 29.2% validation used the previous hand behavior and are not comparable current-hand scores; the unchanged prior failed both current-hand controls tested here.

## Final execution checks and integrity

The final v59 folder was executed on orange11 seed0: normal completion, no harness/sandbox error, but the lift failed (0.3 cm, 42.73 s, $0). This unfavorable replication is preserved separately from the primary 48. The separate egg00 seed0 final-folder control succeeded: 8.3 cm, 32.73 s, $0, with no break, hand contact or sandbox block. Both smoke outcomes are retained outside primary metrics. A fresh wrap-up execution requested by the user also succeeded on egg00 seed0 (job `dev-141421-d6ff`): 8.7 cm lift, 32.4 s, $0, with no break, hand contact or sandbox block. This third execution is also kept outside the primary metrics. An earlier v51 final-folder check succeeded, but it is not claimed as a test of the selected v59 artifact.

No private assets, kitchen models, generator outputs, scoring implementation or held-out identities/files were inspected. Skills use no simulator state or run files. Dev traces were used only for diagnosis; no trace-derived values enter the policy. Only the explicitly permitted whole-body controller was read. Lab code and docs were not modified; the pre-existing AGENTS.md modification was not made by this continuation.

Thirty-five explicit errors from the preceding failed service restart remain recorded separately, with raw outputs untouched. The public lab notice attributes that restart to memory exhaustion and a vision-service restart. `analysis/experiments.json`, `experiment_ledger.csv`, `replicate_results.json`, `excluded_infrastructure.json` and `selected_manifest.json` preserve the evaluation record. All numbered skill folders are retained. Final model spend was $3.3257 of the $10 session cap. All five validation submissions were used. Codex weekly usage was 64.0% against the 66% cap. Source identity and syntax checks passed; the final source SHA256 is `c99d8c739206624abd7a8e8cdbff8718ea56239cc2fb9c77736e5bed31637b27`. Git staging failed because `.git` is read-only (`index.lock`: read-only file system). A second staging attempt failed identically, and the requested escalation failed with `approval request failed`. No commit was created; the preserved version folders and research records are the session record, as allowed by AGENTS.md.
