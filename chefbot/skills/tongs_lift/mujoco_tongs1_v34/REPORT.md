# Tongs skill research report — wrap-up, runner blocked

**The task is not solved.** No complete food lift has succeeded in the completed development episodes. The best acquisition experiment held, reoriented and reopened the tool in all 8 episodes across four development scenes and two seeds, covering pivot, molded, scissor and spring-band tongs. That milestone must not be confused with food-lift success.


**Selected artifact:** `skills/final/main.py`, byte-identical to v34, the most broadly evaluated full-task reference. This selection is not evidence of an effective food-lifting skill: its completed development results are **0/18 successes, 15/18 clear tool holds, 3 breaks, and no hand–food contact**. All full-task versions have zero observed successes. Unrun candidates were not promoted. Mean episode time is **41.06 s** and mean model cost is approximately **$0.00039** for these failed attempts; successful-task time and held-out performance are unknown.

At the 37.9-minute final wrap-up checkpoint, the required `./lab run skills/final --scene dev_00_egg --seed 0 --no-video` check again failed before startup with `BrokenProcessPool`. The preceding v46 experiment also failed before startup. The final file parses and matches the tested reference, but its required smoke run is **not completed**. Validation also has no aggregate result; one earlier failed submission consumed a slot. The remaining four slots were preserved. The task remains unsolved and the program is not ready for real-robot use.

## Approach

The program first parks the hand in front of and below the counter to clear the pole camera's view. A semantic model identifies the requested food and the tongs' joined end and two tips. Grounding DINO tool boxes and the semantic landmarks are matched jointly, including a check for transposed coordinates; SAM supplies tool and food masks. Calibrated depth estimates the support plane, the tool axis and handle cross-section, and the food center. Bounded sphere and ellipsoid fits reduce partial-surface food-center bias.

The base moves closer and aligns laterally for acquisition. The hand closes across the handles, lifts and retracts the tongs, and changes orientation for use. Experiments compare vertical, tilted and horizontal tool poses, different acquisition closures, depth-based held-tip calibration, and spring-width feedback. The current broad reference, v34, uses a horizontal tool with a fixed small closure increment. Prepared v36–v38 replace this with incremental resistance-limited closure, a slower approach, and a paddle thickness-center correction; these versions have NOT run successfully through the lab interface yet.

The supplied prior contributed calibrated depth reconstruction, support estimation, PCA, bounded surface fits and motion convergence patterns. Its robot-specific dimensions and fallback closing angle were not copied: this lab's 0.103 m pad offset and +45-degree closing direction are used. A visible-finger depth diagnostic independently confirmed the closing direction on two scenes.

## Completed development evidence

Every version is preserved under `skills/vN`. Experiment details, including rejected hypotheses and safety failures, are in `research_log.md`. `analysis/dev_summary.json` summarizes completed development artifacts only.

Across the entire local experiment inventory there are 106 completed dev result artifacts, zero complete successes, 14 target breaks and one forbidden hand–food contact. This includes hover-only and acquisition-only experiments, perception failures and repeated versions on the same cases; it is not a set of independent full-task trials or a generalization estimate.

| Version / experiment | Cases | Complete success | Tool held clear at end | Target breaks | Mean world time |
|---|---:|---:|---:|---:|---:|
| Example, hover only | 2 | 0/2 | 0/2 | 0 | 4.24 s |
| v4, first partial grasp | 4 | 0/4 | 1/4 | 0 | 20.70 s |
| v10, acquire/turn/reopen only | 4 | 0/4 | 4/4 | 0 | 27.00 s |
| v12, acquire/turn/reopen only | 8 | 0/8 | 8/8 | 0 | 48.03 s |
| v24, horizontal food attempt | 2 | 0/2 | 2/2 | 1 | 36.74 s |
| v30, softer pickup | 2 | 0/2 | 2/2 | 0 | 35.30 s |
| v34, broad cheap-model reference, incomplete batch | 18 | 0/18 | 15/18 | 3 | 41.06 s |

The v34 batch requested 24 scenes at seed 0, but only 18 have completed artifacts. The other six are uncompleted, not scored as manipulation failures. All recorded success counts come from `result.json`, not the skill's returned dictionary. A held tool alone does not imply controlled handles, correct tip placement, or food possession. The sensor-only pickup audit also exposes an unchecked acquisition failure: the kiwi dev_05 attempt closes with 338 mm flange-position error and essentially zero hand opening, then continues toward the food. Its ik_ok flag is still true. Sixteen other cases fall within the approximate initial lateral-clearance bounds, which does not certify their actual contact configuration.

Significant negative results:

- Some arm orientations stalled 6–10 cm short of their requested food poses. Horizontal use improved tracking but did not produce a lift.
- A detector's highest-scoring box could be a pan handle. v11's incorrect selection caused one target break and one forbidden hand–food contact. Joint semantic/box matching fixed that observed selection failure in the next acquisition comparison.
- A held-tool language-model call confused the hinge with a second tip. Its proposed corrections were rejected by the geometric check.
- Fully opening the hand in one diagnostic still left the measured opening restricted to roughly 4–4.5 cm with the tool retained. The exact contact configuration is unresolved; claims about a secure outside grasp would be premature.
- Nearer-hinge and some other approaches broke food on first tong contact during descent, before the intended squeeze. This motivates gentle approach control and better aperture/placement verification, not stronger blind closure.
- The code review found that the old tool offset uses a visible paddle top while the food target is a body center. v38 corrects this estimated thickness bias, but there is no experimental evidence for the correction yet.

Additional offline audits of the completed v34 episodes found tong–food contact in only 8/18 episodes; 10/18 never made contact. All three recorded breaks occurred before the final food approach finished, before the intended closure stage. More specifically, the peach breaks during transfer toward the hover pose, while the apricot and banana break during final descent. Slowing only final descent does not directly address the peach failure phase. The [contact timelines](analysis/reference_contacts.pdf) show the recorded contact flags and stage endpoints for all 18 completed cases. Sensor logs also show measured reopening widths more than 5 mm below the commanded unloaded width in 11/18 cases. In ten restricted-opening cases with unchanged command, continuous finger contact and no food contact during the move to hover, the deficit remains above 5 mm after another 30–100 steps; median opening increases only 0.668 mm. This weakens a brief settling-transient explanation, without proving equilibrium or identifying the contact mechanism. These observations motivate direct aperture measurement and gentle approach; no trace-derived values were added to the skill. The trace object-position versus fitted food-center comparison (median 5.6 mm, maximum 21.4 mm) is diagnostic only because object origins need not be grasp centers.

Prepared v40 changes only v38's approach opening to full open (g=0), following v25's opening-only tool-retention observation. It has compiled but has not run and is not the final selection.

Prepared v41 adds direct open-head depth calibration to v40 without another model call. Four geometry-only tests pass, including 100 rigidly transformed analytic point sets. Real occlusion and task performance remain untested because the runner is unavailable.

The v22 compliance audit found that its tomato closure stopped while measured opening was still 2.47 mm below nominal unloaded opening. Exceeding an earlier empty-tool curve alone therefore did not establish positive closing preload under the documented spring approximation. v36 retains a positive residual requirement, but still needs testing to separate tool resistance from food loading. In v34, 11 of 15 cases with finger–tong contact at the final squeeze still measure more than 1 mm below the commanded nominal unloaded hand width. The sole case with both intact food contact and finger–tong contact then has approximately zero residual; the two positive-residual cases lack food contact. Thus the fixed closure increment does not establish food-holding preload.

A sensor-log-only reprojection audit provides another visibility warning: the predicted tips in v34 project behind the wrist housing in the two primary pole views. Separate diagnostic camera fits reproduced withheld survey points within 0.002 pixels; no fitted camera constants enter any skill. v42 changes v41 to a sideways horizontal tool orientation to test improved visibility. It remains untested.

An additional analytic counterexample exposed a v41/v42 failure: two visible edges of one paddle can masquerade as two heads, shifting the inferred center by about 30 mm. Prepared v43 compares the apparent head spacing against the initial depth measurement from the same episode. It rejects that counterexample and passes six geometry checks, including 100 transformed two-head fixtures. This is not a demonstrated perception or manipulation improvement: real visibility remains untested, plausible-span false positives remain possible, and rejecting a correction does not verify the original offset. The runner rejected its attempted episode before startup.

A debugging-only rigid-offset audit found that, during reopening, the traced tong object origin moves relative to the sensor-logged flange by a median 11.57 mm. Eleven of 18 cases exceed 5 mm even after allowing for endpoint rotation errors and coordinate rounding; ten of those still have finger contact at both endpoints. This is not a tip-error measurement or proof of slipping: internal articulation can move the traced origin. It reinforces the need to reobserve the open tool. All trace calculations remain in analysis/audit_tool_rigidity.py and its output, with no calibration values transferred to a skill.

The same audit finds smaller changes while the tool remains closed: medians of 2.80 mm during acquisition and 4.45 mm during turning. Only zero and one cases, respectively, exceed the 5 mm lower bound while retaining finger contact at both endpoints. This supports prioritizing reopening and perception over another initial-grasp change. Rotation about the traced object origin remains unobservable in this audit.

The sensor-only food-fit audit identifies another limitation: the reference closes along world Y regardless of food shape. Four completed cases have segmented visible Y extents above 80 mm, and three use the visible bounding-box midpoint after bounded shape fits fail. A food may fit the tongs along another direction. These observations motivate shape-dependent orientation and clearance verification; the saved logs cannot establish a feasible direction or actual aperture. No object-specific size or orientation table was introduced.

Prepared v44 adds diagnostic sensor recording to v34, preserving its commanded motions. It logs millimeter depth, camera calibration and synchronized robot state for the initial survey and opened-tool views so future geometry analysis can use actual observations. Encoding and invalid-depth checks pass; its episode could not start. Logging overhead is unknown, and this diagnostic is not selected as final. The prepared sensor-only analysis script produces calibrated food-mask checks, visible widths and opened-tool prediction overlays; three camera/transform arithmetic checks pass, but it has no live recording to analyze yet.

Prepared v45 adds shape-dependent horizontal approach orientation to v38. It uses accepted fitted-shape widths, or a conservative elongated-visible-cloud fallback, and requires a 20% estimated width reduction within a ±60-degree yaw bound. Six analytic geometry checks pass; its live attempt could not start. Actual clearance, reach and collision behavior remain untested, so it is not the final selection.

A later [landmark image review](analysis/v34_landmark_contact_sheet.png) found three apparent initial end-assignment failures in v34: the lime spring-band tool and two molded tools (apricot and potato). A free paddle is labeled as the hinge and the tip marks are placed at the joined-end side; the apricot's joined end is outside the image. On the four scene/seed pairs shared with v12, three initial grasp geometries are nearly identical, but the lime axis is reversed by approximately 180 degrees. The earlier Robotics-ER lime labels agree with the visible paddles and joined end. This is diagnostic image review, not a controlled model-performance comparison: v12 and v34 also differ in tool orientation and subsequent actions. Prepared v46 changes only v34's initial model selection to prefer Robotics-ER. Its first paired-trial attempt also failed before startup, so no improvement is established and it is not the final selection.

## Time, model cost and validation

The cheap initial Gemini model costs about **$0.0004 per episode** on completed broad-reference cases, with one semantic call. Detection and segmentation have no model-dollar charge but do consume time. v34 averages 41.06 world seconds across its 18 completed cases (median 32.54 s, range 26.60–76.67 s). All delays are included; none have been removed as outliers. Earlier acquisition-only v12 has substantial initial delays and a 48.03 s mean despite fewer actions.

The [timing audit](analysis/reference_timing_audit.json) partitions v34's mean into 7.31 s before the first recorded proprioception call, 3.70 s through the initial park endpoint, 5.03 s from park to completed initial geometry, and 25.02 s thereafter. Four cases already have more than 10 s of elapsed world time at their first proprioception call; the records do not establish the cause. Initial perception has its own tail: park-to-geometry median 2.03 s, maximum 29.27 s. Reported model time averages 3.07 s across the whole episode, so neither the initial delay nor the entire perception interval should be attributed to model latency. The post-geometry interval includes observations and computation as well as motion. These are descriptive boundaries, not measured or guaranteed speedups; successful-task runtime remains unknown because no complete task succeeded.

One validation submission was attempted on the unchanged provisional final (v34) after the prolonged runner outage, to assess generalization and check whether the public validation path was available. It also returned BrokenProcessPool before producing any aggregate result. The attempt consumed a submission: four remain, and validation history is empty. Earlier validation had been deferred pending a full-task development improvement; no validation result has been used for tuning. `skills/final` currently contains the tested v34 reference as a provisional artifact (0/18 completed dev successes, 15/18 clear tool holds, 3 breaks). No final-skill validation result exists. The required final smoke command was attempted, but returned `BrokenProcessPool` before an episode could run; it is NOT a passed smoke test.

Completed v34 cases, grouped using the public dev-scene metadata. These small subgroups describe the observed reference failures; they are not estimates of generalization.

| Container / support | Cases | Complete successes | Clear tool holds at end | Breaks |
|---|---:|---:|---:|---:|
| bowl | 4 | 0 | 2 | 0 |
| counter | 6 | 0 | 6 | 1 |
| pan | 4 | 0 | 4 | 1 |
| plate | 4 | 0 | 3 | 1 |

| Tongs mechanism | Cases | Complete successes | Clear tool holds at end | Breaks |
|---|---:|---:|---:|---:|
| molded flexure | 4 | 0 | 4 | 1 |
| pivot | 11 | 0 | 10 | 2 |
| scissor | 2 | 0 | 0 | 0 |
| spring band | 1 | 0 | 1 | 0 |

The three broken targets were the banana on the counter, apricot in the pan, and peach on the plate. Both tested scissor-tool cases lacked a clear tool hold at the end. No case in any subgroup achieved the complete task.

The [retention-stage audit](analysis/retention_stage_audit.json) distinguishes those tool losses. The bowl kiwi never makes finger–tong contact. The bowl strawberry retains contact through the turned-tool endpoint but loses it during reopening, with no later recontact. The plate peach retains contact through reopening, becomes intermittent during transfer toward the food, and has brief later recontacts through closure before ending without the tool. Thus checking only initial acquisition would miss two observed retention failures. These are contact timings, not identification of a release mechanism or proof of stable earlier grasps; no mechanism-specific controller rule was introduced.

## Known limitations and real-robot transfer

The broad reference fails on all completed food tasks. It retains most pivot and molded tools but also loses or fails to retain some scissor/bowl and kiwi/bowl cases. Food placement and aperture are not verified well enough. Containers add rim collisions, and fragile foods can break on approach. Actual tool motion within the hand and the partially occluded tips remain insufficiently observed. The wrist camera frequently misses the tips in the tested poses. Under a rigid grasp, changing the whole-hand world pose cannot bring an out-of-frame tip into the hand-mounted wrist camera: their relative transform is unchanged. The prepared posture change therefore targets pole-camera visibility; wrist visibility would require changed relative tool geometry or removal of an external occluder.

The [motion endpoint audit](analysis/motion_endpoint_audit.json) also exposes unchecked motion failures. Using the controller's existing 3 mm / 0.04 rad tolerances, only 8/18 pre-closure poses converge, compared with 18/18 turned and opened poses and 14/18 food-approach poses. Reaching the commanded hand pose does not verify the tool pose or acquisition. Conversely, continuing after a missed pose can invalidate the planned grasp. Recovery needs explicit observation; merely extending a blocked downward motion could increase contact loads.

The architecture uses documented RGB-D, proprioception and motion APIs and is conceptually portable, but the present program is **not ready for real-robot use**. It needs experimentally demonstrated controlled tool grasping, reliable tip and aperture tracking, gentle contact detection, safe recovery and food-possession verification. The API's indirect gripper spring estimate is not a substitute for validating contact forces on hardware.

## Integrity and external blocker

No private kitchens, object models, generators, validation/test identities, scoring implementation or simulator environments were inspected. The lab and docs were not modified. The explicitly permitted whole-body controller was read. Selected development traces were used only to diagnose contact timing and displacement; no trace-derived dimensions, forces or object tables were placed in the skill. Completed episodes report zero sandbox blocks.

At about 11:39 local, the lab began returning `BrokenProcessPool` for new runs and existing-job status. A serial retry also failed. Budget queries still work. The user was asked for a supported recovery command or service restart. I have not attempted to alter or restart private lab internals.

A resumed-session serial retry at approximately 302 minutes remaining still failed before episode creation. `jobs` is readable, but its completed-future counts are not completed episode results: status for both interrupted jobs still raises BrokenProcessPool. Prepared v39 adds sensor-only held-tool prediction overlays and segmented-depth diagnostics after pickup and opening. It passes compilation and has no experimental result.

The external helper `analysis/resume_experiments.py` retries the next unstarted episode once per minute using only `./lab` while its execution session remains active. It cannot be relied upon to continue after a turn ends. Following the landmark audit, it now prioritizes three v46 comparisons (dev_07, dev_02 and dev_14, seed0) and stops for analysis. The v44 held-tool recordings remain pending afterward, using the perception choice supported by the comparison. The helper also stops at 45 minutes remaining or on a different error. Queue state is separate from experimental results; no helper is currently running.

After eight retries during one active turn and a separate v38 request, the same pre-episode error persisted. The research-owned retry helper was stopped; no background experiment queue is currently running. The lab service itself was not stopped or modified.

At the latest checkpoint (37.9 research minutes remaining), v46/dev_07_lime/seed0 still fails before startup with BrokenProcessPool, including with video disabled. All three queued v46 comparisons and all three pending v44 recording scenes have been attempted and failed before startup. Model expenditure remains $0.4269, four validation submissions remain, and validation history is empty. No research-owned retry helper is running. The provisional final is byte-identical to the tested v34 reference; v46 remains separate and untested. Confirm recovery with a successful dev episode before spending another validation submission.

**Uncompleted work:** successful execution of the required final smoke run and an aggregate validation result remain blocked by the runner failure. Future research needs supported worker recovery, the pending model comparison and sensor recordings, and demonstrated full food-lift success before any improved selection is justified. The submitted final artifact is v34; this report makes no success claim.

The [artifact manifest](analysis/final_artifact_manifest.json) records the selected file hash, existing reference results, uncompleted checks and pending comparison at the latest packaging checkpoint. It contains no new experimental results.

## Commit status

The requested commit could not be created. `git add REPORT.md research_log.md analysis skills` failed because the sandbox mounts `.git` read-only and could not create `index.lock`. An elevated staging request then failed with `approval request failed`. All deliverables and version folders remain in the workspace; none was committed in this wrap-up. This permissions failure is separate from the lab worker outage.
