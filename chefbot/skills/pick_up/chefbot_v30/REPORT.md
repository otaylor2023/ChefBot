# Pick skill research report — interim

Final selection is still in progress. The validated reference is v30; v36 has completed 191/192 development successes and is awaiting aggregate validation. Distinct perception candidates are still being compared. This report will be updated with the selected `skills/final`, its exact results, and its required smoke test.

## Approach

The program uses only the documented chefbot-v1 robot API and arm motion. It detects the requested object with Grounding DINO, obtains a semantic body point from the cheapest available Gemini model, and segments a compatible object box with SAM 2. Later candidates filter compatible boxes by relative detection confidence before choosing a whole-object box, and reject implausibly broad masks before motion.

Calibrated wrist depth supplies all task geometry. A local height histogram estimates the counter, and robust point-cloud bounds and horizontal PCA estimate the object. Small objects permit a search for a closing direction close to the initial wrist pose, subject to a measured 60 mm width limit. Wider objects retain the narrow direction. Bounded sphere and ellipsoid fits can correct the center of compact rounded surfaces; neither object names nor category-specific dimensions affect grasp geometry.

The hand frame required special attention. The brief specifies a −45° closing direction, but visible open-finger depth consistently showed an approximately +45° line in the supplied flange frame. The initial fixed −45° implementation failed on many elongated objects. A +45° diagnostic recovered those cases. The later program estimates the line directly from wrist depth in the known finger region, accepting only a sufficiently open hand, many points, a plausible spread, and strong covariance anisotropy. It retains the documented direction as fallback. This is sensor calibration, not a simulator-state lookup.

Top-down grasps first lower the hand at its initial horizontal position. Tall objects use a 45° tilted body grasp with a short axial insertion. A blocked top-down hover or descent can trigger a 30° tilt that preserves the intended pad midpoint. The hand closes, then requests a 13 cm lift. The latest candidates use shorter motion batches with tighter feedback checks; a held object whose lift stalls can be retracted 5 cm toward the base before lifting again. All motions keep the base commanded stationary through `act_arm`.

## Development evidence

Development has 24 object/kitchen scenes. Seeds vary placements and orientations; additional seeds do not create independent object categories. Matched comparisons and all version folders are recorded in `research_log.md`. Machine-readable completed results are in `analysis/evaluations.json`.

| Version | Development cases | Successes | Mean world time |
|---|---:|---:|---:|
| Example, hover only | 2 | 0 | 10.64 s |
| v1, initial hand mapping | 24 | 14 | 11.03 s |
| v3, corrected hand mapping | 48 | 41 | 10.52 s |
| v10, first integrated picker | 96 | 89 | 12.95 s |
| v22, geometry and reach improvements; seeds 0–5 | 144 | 141 | 13.15 s |
| v29, faster feedback and mask/lift fixes; seeds 0–5 | 144 | 143 | 10.90 s |
| v30, width invariant; seeds 0,1,4–7 | 144 | 140 | 10.93 s |
| v36, shallow fallback and bounded box expansion; seeds 0–7 | 192 | 191 | 11.87 s |

V36 passes 191/192 placements, with a 10.87 s median and one cheese grasp that lifts but is not scored held. On the 144 cases shared with v30, it improves 140/144 to 143/144. Its difficult can lift and both shallow approach recoveries actually activate and succeed. These counts come from scored results, not the skill’s gripper-width return value.

V22 passed 96/96 cases on seeds 0–3, averaging 13.08 s, then 45/48 on fresh seeds 4–5, averaging 13.30 s. Its three fresh failures were two background masks selected from oversized low-confidence boxes and one held can that could not complete its lift. Every v22 episode finished within 22.8 s. Its returned `ok` used gripper width as a proxy; the can failure demonstrates that a nonempty hand alone does not establish task success. All success numbers here come from the lab's scored results, not return values.

Selected controlled experiments:

- Correcting the closing-axis mapping improved a matched five-case test from 1/5 to 4/5. The full seed-0 comparison improved from 14/24 to 20/24.
- A strongly retracted 60° tilted approach often failed IK. A 45° approach improved the six tall cases to 5/6; short axial insertion reached 6/6.
- Wrist half-turn recovery and rotating before translation each recovered only one of five blocked cases. Their added time did not justify broad adoption.
- Feasible closing-direction search plus conditional tilt reached 17/18 on a difficult subset, versus 12/18 for v10.
- Compact-surface center correction reached 20/20 on its test. Using the fitted ellipsoid's full grasp cross-section regressed to 18/20 and was rejected. One successful wide-object lift remained marginal at 6.4 cm.
- Shorter feedback-controlled motion batches preserved 16/16 successes while reducing mean time from 13.60 to 11.59 s on a matched subset.
- Confidence-filtered boxes recovered both observed broad-mask failures in a 16/16 test. A geometric mask guard alone reached 5/6 versus 4/6; point-only SAM could still select an object part.
- Uprighting a stalled held grasp failed in three repeated trials of one can placement. Horizontal retraction followed by lifting succeeded in three repeats with actual recovery activation. These repeats are not three independent placements.

## Held-out validation

Only aggregate output from the permitted validation interface was read. No validation identities, images, traces, or result files were inspected.

| Version | Successes | Mean world time | Mean model cost |
|---|---:|---:|---:|
| v10 | 17/24 | 12.9854 s | $0.0003 |
| v22 | 20/24 | 13.2917 s | $0.0003 |
| v30 | 20/24 | 11.4383 s | $0.0003 |

The development-to-validation gap is material. Development success on repeated categories does not establish the same success rate on unseen categories. Later candidate validation is pending.

## Time, cost, and limitations

V36's 192-case development mean is 11.8657 world seconds and $0.00031 per episode; median 10.87 s, maximum 86.27 s. Three episodes exceed 30 s (46.8, 60.93, 86.27 s), with 37.1, 49.88, and 77.56 s respectively recorded in model-related calls. All three succeed, and all are retained in the mean. The call log identifies Gemini 3.1 Flash-Lite; the source chooses the cheapest available Gemini dynamically. Time includes inference and computation while the world continues running. Detection and segmentation are free in model dollars but consume time. There is one cheap semantic model call per normal attempt. The shorter motion-feedback batches account for the main measured speed improvement; service waiting remains an uncontrolled timing tail.

Remaining risks include wrong semantic identity, SAM selecting an object part or background, partial-depth center bias, near-aperture objects, tipping, slipping, collision-blocked approaches, and IK dead ends. Width and flange motion cannot prove target identity or secure pad contact. The skill has no final visual possession check. Its current improvements do not establish a universal 30-second guarantee.

The architecture is transferable in principle: it uses RGB-D calibration, joint/flange observations, and ordinary Cartesian arm commands. The runtime finger-axis estimate directly addresses a frame discrepancy without relying on private robot models. It is not ready for unattended hardware deployment. Real depth holes, reflective or transparent packaging, uncertain friction, calibration drift, and safe contact-force/collision handling require hardware testing. The supplied API provides no force/torque feedback or full arm collision model; recovering from simulation tracking stalls is not a hardware safety certification.

## Provenance and integrity

The supplied prior skill contributed semantic-point/box matching, DINO/SAM perception, support estimation, PCA, and bounded convergence ideas. Its YAM flange frame, 94 mm opening, stool assumptions, camera choice, and reported success rate were not transferred as facts about this Panda task. The provided source and report had differing version provenance, recorded in the research log. This work added Panda-specific sensor calibration, width-constrained headings, revised tilted insertion, compact-surface fitting, and new feedback/recovery experiments.

No simulator environments, object models, private scenes, validation/test assets, hidden state, object/category lookup tables, or base motion were used. No trace files were read. The public CLI, provided documentation/prior work, and explicitly permitted controller file were read; the lab was not modified. All completed evaluations so far report zero sandbox blocks. Source versions are preserved in separate `skills/vN` directories and lab snapshots. Git metadata is read-only in this workspace, so those folders and snapshots provide the version record.

Final artifact, exact final benchmark, validation selection, and smoke test: pending completion of the ongoing experiments.
