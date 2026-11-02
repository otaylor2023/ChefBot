# Tongs lab research report

The selected program is [skills/final/main.py](skills/final/main.py), currently byte-identical to v17. It achieves **14/48 successes on the two primary development seeds** and **29.2% success on 24 held-out validation episodes**. Validation averages **44.01 world seconds and $0.0043 per episode**, including model latency. This demonstrates limited generalization, but it is not a reliable cooking skill or ready for unsupervised hardware use.

The selection uses completed evidence. Unfinished comparisons and a second final-artifact smoke check are listed below; none has been promoted on partial results.

## Approach and prior reuse

The supplied prior v34 became this session's v1 unchanged. I retained its survey pose, calibrated RGB-D reconstruction, local support estimate, tool PCA and cross-section geometry, bounded food surface fits, and whole-body motion helper. The supplied prior reported 0/18 complete food lifts; our unchanged-reference check scored 0/2 with both tools retained. These are different evaluation samples and are not pooled.

The selected program:

1. Parks the hand to expose the pole-camera view. One Robotics-ER call identifies the requested food and the tongs' joined end and two paddles.
2. Uses Grounding DINO and SAM masks. Food detections are matched to the semantic point, preventing a higher-scoring neighbor box from overriding the requested target.
3. Estimates the support, tool axis, handle width and a grasp at 45% of sensed tool length. It uses the documented 0.103 m fingertip offset and +45-degree closing line, and moves the base closer and laterally toward the grasp.
4. Picks up the tongs and turns them horizontally at 60-degree azimuth. A depth fit measures the open paddles relative to the actual flange pose.
5. Records an empty-tool hand-width versus closure-command curve. This supplies an in-episode reference for tool resistance.
6. Moves above the food and descends in 1.2 mm increments to 8 mm above the fitted food center. It closes in small increments until additional resistance exceeds the empty curve, with positive nominal spring residual.
7. Requests a 9.5 cm lift with slight retraction, then returns an explicitly **unverified** lift attempt.

The program uses only the documented robot API. It has no per-object size, grasp, force or prompt table. The returned dictionary does not verify possession; reported successes below are the lab's physical outcomes, not claims made by `run()`.

## Selected-skill results

| Evaluation | Episodes | Successes | Tools retained | Broken foods | Hand–food contacts | Mean world time |
|---|---:|---:|---:|---:|---:|---:|
| Dev seed 0 | 24 | 8 | 19 | 1 | 0 | 44.37 s |
| Dev seed 1 | 24 | 6 | 18 | 2 | 0 | 45.89 s |
| Combined primary dev | 48 | 14 | 37 | 3 | 0 | 45.13 s |
| Held-out validation | 24 | 29.2% | Not provided | Not provided | Not provided | 44.01 s |

The validation job is `val-175203-a255`, digest `237fe3ce564e`. Its public aggregate reports zero harness errors and zero sandbox blocks. Only that aggregate was accessed; no validation identities, files or per-scene outcomes were inspected. Three validation submissions remain. A second public aggregate is pending for the distinct semantic visual-closing approach v54 (`val-215452-815d`); it was submitted as an approach comparison, with no claimed dev-count improvement.

Combined dev model cost is **$0.00393 per attempt**. Dev runtime median is 44.37 s, 90th percentile 50.01 s and maximum 53.87 s. Successful dev attempts average 45.26 s. Thus the selected program stays below the 90 s limit in these trials but exceeds the 40 s target on average. These are observed attempt costs, not a guarantee of successful-task completion time.

The primary evaluation excludes extra repeats and smoke checks:

- `skills/final`, dev00 egg seed0 (`dev-174229-f0ea`): execution completed, but manipulation **failed**; 0 cm lift, intact food, tool retained, 41.2 s.
- Extra v17 dev00 seed2 (`dev-173542-3b0b`): success, 8.9 cm lift, 50.87 s.
- Second final-artifact check, dev03 apricot seed0 (`dev-204306-022f`): **pending**. This is a previously successful control, not an unbiased evaluation sample.

### Dev failure groups

These groups combine both primary seeds and describe development outcomes only.

| Support | Episodes | Successes | Tools retained | Broken |
|---|---:|---:|---:|---:|
| Counter | 16 | 7 | 14 | 2 |
| Pan | 12 | 3 | 11 | 1 |
| Plate | 8 | 4 | 6 | 0 |
| Bowl | 12 | 0 | 6 | 0 |

| Mechanism | Episodes | Successes | Tools retained |
|---|---:|---:|---:|
| Pivot | 32 | 10 | 29 |
| Molded flexure | 8 | 2 | 4 |
| Spring band | 4 | 2 | 4 |
| Scissor | 4 | 0 | 0 |

Within pivot designs, the public bare-steel/bare-handle style succeeds in 3/8 trials (8 tools retained), studded soft-grip paddles in 3/8 (8 retained), ribbed soft-grip paddles in 4/12 (9 retained), and nylon paddles in 0/4 (4 retained, one break). These small groups mix different foods and containers; they do not isolate a material effect. Full descriptions and counts are in [the style breakdown](analysis/v17_style_groups.json).

Eggs succeed in 3/4 episodes and apricots in 6/8. The selected baseline has no tomato, onion or lemon successes in these two seeds. The three breaks are dev06 banana seed0, dev21 banana seed1 and dev20 pan-apricot seed1; all occur before the closing loop.

## What worked and what did not

Robotics-ER corrected observed end-assignment errors from the cheaper initial model. Explicit y,x coordinates and semantic matching of food boxes reduced perception errors. The earlier highest-score food-box policy caused one forbidden hand–onion contact while approaching a neighbor; this failure is retained in the record.

Empty-tool calibration enabled lifts with bare-steel tongs where absolute hand resistance was insufficient to identify food loading. The full seed-0 comparison improved from **v14 6/24 to v17 8/24** after adding the approach-height margin. Four pairs improved and two regressed. This small, variable comparison supports the selection, not a statistically established universal improvement.

The broadest alternative, v45, selects opening from measured tip aperture and fitted food width. It completed the same 48 pairs with **14 successes, 37 tools retained, four breaks, no hand contact, 44.01 s mean and $0.003905 mean cost**. Its preliminary gain disappeared in the complete comparison. It is slightly faster but has no success-count gain and an additional break; v17 remains the better-established selection. All four v45 breaks also precede closure.

Other completed comparisons illustrate the limitations:

| Change | Successes | Matched reference | Finding |
|---|---:|---:|---|
| Visual probe and retry, v25 | 2/5 | v17 2/5 | About 52 s; a successful small probe did not prevent later slip |
| Support-based contact height, v29 | 1/8 | v17 3/8 | No improvement |
| Two-view food fusion, v32 | 0/5 | v17 1/5 | Fitted centers changed little |
| Lift-time load maintenance, v37 | 1/7 | v17 3/7 | One broken egg; break preceded substantial added closure |
| Slow pickup, v38 | 2/7 | v17 3/7 | Pickup tracking errors persisted |
| Controlled first reopening, v39 | 2/11 | v17 4/11 | Two breaks; preload alone left inadequate tip aperture |
| Narrower sensed handle grasp, v41 | 0/7 | v17 3/7 | No retention or success gain |
| Higher pickup pad alignment, v44 | 2/7 | v17 3/7 | Fewer tools retained |
| Rolled closing line, v46 | 0/8 | v17 3/8 | One tomato broke during roll/settling, before closure |
| Segmented visual closing check, v47 | 3/8 | v17 3/8 | Never authorized extra closure; head fits usually failed |
| Post-transfer head refresh, v48 | 2/8 | v17 3/8 | No demonstrated gain |
| Staged horizontal calibration/vertical use, v49 | 1/4 | v17 1/4 | No bowl success; two large orientation misses |
| Full-aperture handle preload, v50 initial sample | 4/8 | v17 4/8 | Same successful cases; broader replication pending |
| Higher closing-stop load, v52 | 3/8 | v17 3/8 | Gains tomato/onion, loses egg/lime controls |
| Alternate wrist polarity for v49, v53 | 0/4 | v49 1/4 | No improvement |

The vertical families repeatedly encountered large pose errors, occlusion or contact during entry. Their failures do not establish that a correctly placed vertical grasp cannot work. Semantic grasp selection also needs a feasible-width constraint: one scissor attempt selected an 85 mm section for an 83 mm hand; another ended on malformed model output before pickup.

The targeted closing-model prototype v51 scored 0/8 because a 300-token allowance truncated its JSON and the parser exited. These are skill failures, not infrastructure exclusions. v54 increases the allowance and handles malformed responses; seven simulator-free fallback checks pass. Its robot comparison completed 2/8 versus v17 3/8, with all eight tools retained, no breaks or hand contacts, and 58.78 s mean time. Paddle points sometimes guided extra closure, but also pointed at food, and model latency was substantial. It does not improve the completed dev comparison. In one egg regression, contact became intermittent and largely disappeared during an 8.4 s visual-check interval; earlier contact histories also differed, so delay alone is not an isolated causal explanation. Its aggregate validation was queued before the final three dev results arrived and remains pending.

## Limitations and transfer

The selected policy fails all tested bowls and scissor episodes. Horizontal shafts can meet rims; changed postures have not solved placement reliably. Other limitations include partial-view food fits, uncertain aperture, unstable contact configuration, slips during lifting, and imperfect detection of tool possession. Hand-width resistance cannot uniquely distinguish tool spring/friction, food loading and other contacts. The failed final-folder egg repeat had identical initial geometry to its successful primary control, but its hand opening increased during approach and it stopped closing much earlier without food contact. This exposes sensitivity to the subsequent grasp/contact history. The saved end views show the [successful primary attempt](runs/dev-171553-617a/dev_00_egg_s0/images/22_food_lift_pole.png) and [failed final-folder repeat](runs/dev-174229-f0ea/dev_00_egg_s0/images/22_food_lift_pole.png).

A full trace audit corrected an earlier coarse inspection: the lemon and onion examples make brief intermittent contacts, including near the closing stop. Sampling every 30 steps had missed those contacts. The correction is recorded in the log; those examples must not be described as never touching food.

The sensor/API architecture could transfer in principle. The current program is **not ready for unsupervised use on the real robot**: it needs demonstrated grip stability, clearance planning, gentle contact control, recovery and possession verification under real depth noise and hardware dynamics. End-of-episode success also does not establish a durable long-term hold. The selected v17 also retains an unguarded initial model-JSON parser and has no general recovery from failed perception; the malformed closing-response guard tested in v54 is not part of the selected program.

## Artifacts, integrity and outstanding work

- [Selected program](skills/final/main.py), SHA-256 `58e62dccdfe37e3a5e43bb47fc4074c5f911e860fc3d2d16c3f971d29d779856`.
- [Research log](research_log.md), [experiment ledger](analysis/experiment_ledger.csv), [paired comparisons](analysis/paired_candidates.json), [primary metrics](analysis/final_metrics_v17.json), and [public validation aggregate](analysis/validation_v17.json).
- Every version is retained in its own `skills/vN` folder. Git metadata is read-only; the folders and lab snapshots preserve the record. The requested staging attempt failed, and its escalation was rejected with “approval request failed”, so no commit was created.

Unfinished evaluations are recorded in [the public pending-job snapshot](analysis/pending_jobs.json):

| Evaluation | Completed / requested | Available outcome |
|---|---:|---|
| v50 full seed-1 dev replication | 13/24 | 4/13 successes, same as matched v17; all 13 foods intact |
| v55 closing-slope feedback | 0/8 | No result |
| v56 feasible semantic grasp | 0/6 | No result |
| v57 combined aperture preload and higher closing load | 0/8 | No result |
| Second final-folder smoke check | 0/1 | No result |
| v54 aggregate validation comparison | 0/24 | No result |

There are 33 unfinished versioned dev episodes, one unfinished smoke check and 24 unfinished validation episodes in that snapshot. They are not counted as failures or successes. v50's 13 partial seed-1 cases plus its eight initial seed-0 cases give 8/21 successes, equal to matched v17; this is not a completed broad evaluation.

Public job status and budget remain responsive during long quiet intervals, with no reported harness error for the unfinished batch. Supported recovery information was requested; no private worker inspection or restart was attempted. Queue elapsed time is distinct from the reported per-episode world time.

No private kitchens, object models, generators, scoring implementation, validation/test identities or simulator environments were inspected. The explicitly permitted whole-body controller was read; the lab and docs were not modified. Dev traces were used only for diagnostic timing and interpretation. No traced positions, dimensions, forces or lookup values enter the skill. Completed versioned attempts audited so far have zero sandbox blocks.

## Final record

The final snapshot at 2026-11-02T06:18:18.614614+00:00 records **397 completed versioned dev attempts** across 57 retained versions; repeats, failed prototypes and skill errors are included in that ledger, while examples and final-folder smoke checks are separate. Total metered model spend is **$1.693 of $10**, with 3 validation submissions remaining. The research budget reports 41.8 minutes left and 58% weekly Codex usage (66% cap). Unfinished queued work is recorded above rather than assigned an outcome.

The [selection manifest](analysis/selected_manifest.json) binds the unchanged final file to the completed v17 evaluations and records the execution checks. [The resource snapshot](analysis/budget_checkpoint.json) and pending-job snapshot are time-stamped. No subsequent result is implicitly included in this report.

A final public status check after that snapshot found unchanged counts and the lab’s 39-minute wrap-up notice. The final selection remains v17.
