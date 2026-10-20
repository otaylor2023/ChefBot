# Ortho-View Grasp Marking — Experiment Spec (v1)

## 1. Question

Can a general-purpose VLM produce a usable grasp by **marking points on orthographic plots** of an isolated object (top view x/y, side view x/z, robot frame, metric grid), instead of outputting end-effector numbers from camera images like current "LLM-as-policy" work (Astranomy, RoboDojo)?

v1 answers this for **one object (a cup), one action chunk**, across a small set of representations, render styles, prompts and models, scored offline, kinematically, and in sim.

## 2. Scope

**In v1**
- Object: cup only (rotationally symmetric, so yaw is free and orientation errors show up as tilt/approach errors).
- Representations P1, P2, B0 (§7), plus two references that use no model (§7.4).
- Output modes O1 (axis meters) and O2 (pixels) (§8).
- Renders V1 (point cloud) full and partial, and V4 (realistic ortho render) (§6).
- Trajectory modes T1 (grasp pose only) and T3 (pre-grasp, grasp, lift) (§9).
- Independent vs sequential view querying (§10).
- Three models (§12).
- Evaluation levels E0–E3 (§14).

**Not in v1:** other objects (mug, bottle, pan — v2, they test orientation), candidate-marker selection (PIVOT-style), discrete approach / numeric-rotation representations, render-and-verify refinement loops, few-shot prompts, obstacles and clutter context in the plots, real robot.

## 3. Where it lives

```
experiments/ortho_grasp/
  configs/            # one YAML per named config (§15)
  prompts/            # versioned prompt templates (§11)
  scenes.py           # seeded cup scenes (§4)
  observe.py          # mask + depth -> object point cloud; full-mesh cloud (§5)
  render/
    plots.py          # V1 point-cloud plots (§6.1)
    ortho_sim.py      # V4 MuJoCo orthographic renders (§6.2)
    camera.py         # B0 perspective camera images + calibration text (§7.3)
    overlay.py        # draw marks + resolved gripper back onto plots (§16)
  query/
    models.py         # model clients (§12)
    schemas.py        # JSON output schemas per representation/output mode
    parse.py          # response -> marks (with failure reasons)
  resolve.py          # marks -> 6-DOF grasp (+ waypoints) (§13)
  evaluate/
    e1_geometric.py
    e2_kinematic.py
    e3_sim.py
  run.py              # run one config over the scene set
  report.py           # results tables + failure gallery (§16)
  results/            # git-ignored
```

Rules:
- **No changes to `chefbot/` core.** Reuse it read-only: the RoboCasa env (spawn the robot at a fixed pose; reuse the ChefKitchen spawner for objects), GT segmentation, cameras, the FR3 IK, the Coal/OMPL collision code from the planning spike, the arm controller and gripper for E3.
- Everything is driven by config files; every axis in §15 is a config field.
- Fully seeded and cached: the same config + scene + repeat index re-uses a cached model response unless `--no-cache`.

## 4. Scenes

- Keep it simple: **the robot is spawned at a fixed base pose facing a counter and never moves.** No `go_to` / `align`.
- One layout (layout 1), one counter segment with clear space in front.
- 30 seeded scenes. Per scene, vary:
  - cup position: sampled in a reachable window in front of the robot (e.g. 0.45–0.70 m forward, ±0.20 m sideways of the arm base, on the counter surface; check IK reachability of a top-down grasp at the cup before accepting the sample);
  - cup model: 2–3 RoboCasa cups of different sizes;
  - distractors on/off: 2–4 other objects within 25 cm (they are removed by the isolation step, so this tests the masking, not the model).
- Record everything needed to replay: seed, cup model, cup pose, distractor poses. The base pose is a constant in the config.
- Each config runs every scene **3 times** (repeat index 0–2, different model sampling), so 90 queries per config.

## 5. Observation pipeline

From the shoulder camera (and wrist camera for fusion where noted), in the robot frame:

1. **Mask:** sim GT segmentation for the cup (stand-in for SAM; VoLo found SAM unreliable on MuJoCo renders).
2. **Partial cloud:** masked depth → back-project → robot frame.
3. **Full cloud:** sample the cup's collision mesh surface (GT, ~20k points).
4. Counter height from GT (used to draw the counter line in side views).

Both clouds are computed for every scene; the config chooses which one is rendered.

## 6. Renderers

All plots share these conventions (a plot spec saved next to each image is the single source of truth):
- **Orthographic, robot frame, equal aspect**, same meters-per-pixel in both views.
- Extent: cup bounding box + 20 cm margin on every side (so a flange point 15.5 cm away fits).
- 1 cm minor grid, 5 cm major grid with labeled ticks in meters; axis names; a small "robot →/←" direction marker.
- A legend panel with a 2F-85 drawn to scale (85 mm max opening, finger length, flange) and the points the model must mark (A, B, F).
- Side view shows the counter as a filled band.
- **Plot spec JSON per image:** view name, world→pixel affine transform (and inverse), extent, image size. O2 parsing and all overlays use it.
- Image size: 1024 × 1024 per view (configurable).

### 6.1 V1 — point cloud plots
- Top view: points colored by height (viridis, colorbar labeled "floor of cup → rim"), drawn low-to-high.
- Side view: points colored by height, drawn far-to-near.
- `geometry: full | partial` selects the cloud.

### 6.2 V4 — realistic orthographic sim render
- Two MuJoCo **orthographic cameras** (MJCF `camera orthographic="true"`, extent set from the plot spec) looking down −z and along +y in the robot frame.
- Render RGB; using the GT segmentation, white out everything except the cup and the counter surface (counter only in the side view).
- Composite the same grid, axes, legend and counter band on top, so only the object appearance differs from V1.
- Verify alignment: project 10 known cup mesh points through the plot spec and check they land on the cup in the render (< 2 px). Fail loudly otherwise.

## 7. Representations

### 7.1 P1 — A, B, F
In each view the model marks fingertip **A**, fingertip **B**, and a point **F** on the flange.
- Resolved (§13): M = (A+B)/2, approach a = normalize(F − M), closing c = Gram-Schmidt(B − A against a), width = |B − A|.

### 7.2 P2 — A, B only
In each view the model marks **A** and **B**. The approach is chosen geometrically (§13.3).

### 7.3 B0 — numeric baseline (no plots)
Astranomy-style, to compare against:
- Input: shoulder RGB + wrist RGB (unmodified camera views), each camera's intrinsics and camera-to-robot extrinsics as text, the current TCP pose, the gripper description.
- Output: TCP position (m) + orientation as a unit quaternion (wxyz) + opening width (m), robot frame.
- Same prompt task wording and same evaluation as the plot conditions.

### 7.4 References (no model)
- **R-GT:** the existing `GroundTruthGraspProposer` rim grasp (upper bound).
- **R-heur:** a dumb heuristic: top-down at the cup centroid at rim height, fingers along x, width 85 mm (expected to fail on most cups; a floor for comparison).

## 8. Output modes (P1, P2)

- **O1 — axis meters:** each mark as world coordinates read off the plot axes, e.g. top view `{"A": [0.497, 0.050]}` (x, y), side view `{"A": [0.497, 0.985]}` (x, z).
- **O2 — pixels:** each mark as `[u, v]` pixel coordinates in that view's image (origin top-left), converted with the plot spec.

The JSON schema is enforced (structured output where the API supports it; otherwise parse and validate). Out-of-range values and missing points are parse failures with a recorded reason.

## 9. Trajectory modes

- **T1 — grasp only:** the model marks the grasp pose. Pre-grasp = grasp − 0.10 m along the approach (backed out), lift = grasp + 0.12 m in +z. Prompt task: **"grasp the cup"** (put the fingers around part of it, ready to close).
- **T3 — three waypoints, one plot:** the model marks a numbered path on the same plots: waypoint 1 pre-grasp, 2 grasp, 3 lift, each with its own A, B, F (P1) or A, B (P2), plus `gripper: open|closed` per waypoint. Prompt task: **"pick up the cup"**. Execution interpolates straight Cartesian lines between waypoints.

## 10. View handling

- **C-ind (independent):** one model call, both views attached, the model returns marks for both.
- **C-seq (sequential):** call 1 gets the top view only and returns top marks. Call 2 gets the side view with a thin vertical guide line at each top-view x (labeled A, B, F, M) and returns z only for each point (x is taken from the top view).
- Conflict policy in C-ind (v1, fixed): **x from the top view**; log the per-point x disagreement. (Weighted / re-query policies are v2.)

## 11. Prompts

`prompts/` holds templates with a version tag stored in every result. Each template has:
- A system part: what the plots are (orthographic, robot frame, units, grid, colors), the gripper legend, what each mark means, output JSON format.
- A task part: T1 "grasp the cup" or T3 "pick up the cup".
- Reasoning: allowed (models may think before answering; final answer must be the JSON block).

Write them once, test on 3 scenes by hand, then freeze for the whole run. Any change bumps the version.

## 12. Models

Available providers: **OpenRouter**, **Gemini API** (Google AI Studio key), **Tinker** (Thinking Machines). One client interface (`query/models.py`) with a backend per provider; model id + provider recorded on every call.

| Id | Model | Provider | Notes |
|---|---|---|---|
| M-claude | current Claude Opus | OpenRouter | default model for the baseline |
| M-gemini-er | Gemini Robotics-ER 2 (`gemini-robotics-er-2-preview`) | Gemini API | pointing-tuned; key comparison for O2 (pixels) |
| M-gpt | GPT-6 Astra if listed, else GPT-5.5 | OpenRouter | |
| M-open (optional) | Qwen3-VL (largest available) | Tinker or OpenRouter | open-weight reference; also the candidate for a v2 fine-tune |

Before the full run, check for each model: images accepted at 1024×1024 (two per call), structured/JSON output works, and **rate limits / quota** cover ~300 calls in a few hours (the Gemini key's tier matters most here; a consumer Gemini subscription does not by itself include API quota).

Fixed per run: temperature (default 0.7 so repeats differ; record it), reasoning effort (record it), max output tokens. Record latency, input/output tokens and cost for every call.

## 13. Resolution (`resolve.py`)

### 13.1 Lifting
For each mark: x and y from the top view; z from the side view (C-ind: side view's own z; C-seq: the z answered for that guide line). Record x disagreement per point (C-ind).

### 13.2 P1 → pose
- M = (A+B)/2; a = normalize(F − M); c0 = normalize(B − A); c = normalize(c0 − (c0·a)a); third axis = a × c; width = |B − A|.
- TCP frame convention: z = approach direction pointing **into** the grasp (i.e. −a, from flange toward fingertips), x = closing direction. Match the repo's TCP convention.
- Record: Gram-Schmidt angle = asin(|c0·a|).

### 13.3 P2 → pose (approach chosen geometrically)
Sample approach directions perpendicular to c every 10°. Keep those where the gripper (finger + palm + body boxes, as in the overlay) does not intersect the object cloud or the counter at the grasp pose and along the 10 cm approach. Choose the one closest to straight down (−z approach); tie-break toward the robot. None valid → resolved as `no_approach`.

### 13.4 Checks (both)
Recorded as flags; a resolved pose may fail several:
- `parallel`: |c0·a| > cos(45°) → rotation undefined (P1 only); **reject**.
- `from_below`: approach points up from beneath (a_z < −0.2 for P1's F direction convention); **reject**.
- `width_over`: width > 0.085 → clamp to 0.085, flag.
- `gs_large`: Gram-Schmidt angle > 10°, flag.
- `x_disagree`: any point's x disagreement > 1 cm, flag.

### 13.5 Opening width used for execution
Pre-grasp opening = min(0.085, width + 0.01). Not fully open: on a rim grasp a fully open gripper hits the far inside wall.

## 14. Evaluation

### E0 — parse
Valid JSON matching the schema, all points present, values inside the plot extent. Report parse-failure rate per config.

### E1 — offline geometric (no sim)
On the **full GT cloud** (even if the model saw the partial one):
- `straddle`: the segment A–B passes through cup material, and the material along it is a single contiguous run (fingers on opposite sides of one wall or of the whole body).
- `clear_tips`: A and B are each > 3 mm from any cup point (fingertips not inside material).
- `width_ok`: material thickness along A–B + 2 × 5 mm ≤ width ≤ 0.085.
- `height_ok`: M is between counter + 1 cm and rim + 1 cm.
- `e1_valid` = all four.
- Continuous metrics: distance from M to the nearest cup surface point; Gram-Schmidt angle; x disagreements.

### E2 — kinematic (no physics)
- `ik_ok`: FR3 IK solves for pre-grasp and grasp from the fixed base pose (repo IK, joint limits).
- `collision_ok`: gripper box model vs object cloud + counter is collision-free at pre-grasp and along the straight approach to grasp (fingers allowed to touch the cup only within the last 1 cm). Use the Coal collision setup from the planning spike.
- `e2_valid` = e1_valid and ik_ok and collision_ok.

### E3 — sim execution (top configs only)
In RoboCasa, from the fixed base pose: move to pre-grasp (joint move), straight line to grasp, close, then lift 12 cm (T1) or follow waypoints (T3).
- `enclosed`: after closing, the cup is between the fingers (GT contacts on both fingers).
- `lifted`: cup bottom ≥ 5 cm above the counter after lift, and still held after 2 s.
- Record unexpected arm/gripper contacts during approach.

### Cost
Latency (s), tokens, $ per query; per config totals.

## 15. Experiment matrix

**Baseline config (`base`)**: P1 · O1 · V1-full · top+side · T1 "grasp the cup" · C-ind · M-claude · prompt v1.

Each set varies one axis from `base` (the axis value in `base` is not re-run):

| Set | Configs | Question |
|---|---|---|
| S1 representation | `p2`, `b0` (+ `r_gt`, `r_heur` references) | does marking beat numeric output; does the flange point help or hurt |
| S2 output mode | `o2` (pixels) | axis readout vs pointing |
| S3 render | `v1_partial`, `v4_real` | does partial geometry or realistic appearance change accuracy |
| S4 view handling | `seq` | does sequential querying fix cross-view disagreement |
| S5 trajectory | `t3` "pick up the cup" | can it draw a whole chunk on one plot |
| S6 models | best-of-S1–S5 config × `M-gemini-er`, `M-gpt`; plus `o2` × `M-gemini-er` | model sensitivity; pointing-tuned model with pixels |

Totals: ~10 model configs × 90 queries ≈ 900 queries (C-seq doubles calls for its config). E0–E2 for all. **E3** for `r_gt`, `base`, and the best two configs: 30 scenes × 1 run each.

"Best" = highest `e2_valid` rate; ties broken by lower Gram-Schmidt angle, then cost.

## 16. Logging, overlays, report

Per query, `results/<config>/<scene>_<repeat>/`:
- `inputs/`: exact images sent, plot specs, prompt text (rendered), config snapshot.
- `response.json`: raw model output, parsed marks, parse errors, latency, tokens, cost.
- `resolved.json`: lifted 3D points, pose, width, waypoints, flags.
- `metrics.json`: E0–E2 (and E3 when run).
- `overlay_top.png`, `overlay_side.png`: the input plots with the model's marks (A, B, F, derived M) and the **resolved gripper** drawn back in green.
- E3: `exec.mp4` (short clip) and contact log.

`results/summary.csv`: one row per query, all metrics and flags.

`report.py` → `results/report.html`:
- Per-set tables: E0/E1/E2/E3 rates with 95% Wilson intervals, mean Gram-Schmidt angle, x disagreement, latency, cost.
- Per-config failure breakdown by flag/metric (which check failed most).
- Gallery: for each config, 6 random queries and the 6 worst, showing both overlays side by side, with metrics; click through to the raw response.
- A scene × config grid (green/red by `e2_valid`) to see whether some scenes are hard for everything.

## 17. Deliverables and acceptance

1. Harness runs any config end to end: `python -m experiments.ortho_grasp.run --config configs/base.yaml [--scenes 0-29] [--repeats 3] [--eval e0,e1,e2]`.
2. Unit tests (no model, no sim): plot-spec round trip (world → pixel → world < 0.5 mm); lifting; P1 resolution on hand-made marks (the four cases from the disagreement drawings: consistent → accept, fingers-vs-flange → `gs_large`, cross-view → `x_disagree`, parallel → `parallel`, from below → `from_below`); E1 metrics on hand-placed grasps around a synthetic cup (valid rim grasp passes; both fingers outside the wall fails `straddle`; fingertip inside wall fails `clear_tips`).
3. `r_gt` scores ≥ 95% `e2_valid` and ≥ 90% `lifted` in E3 (sanity check of the evaluators); `r_heur` scores low.
4. V4 alignment check passes on all 30 scenes.
5. All sets in §15 run; `report.html` generated.
6. A one-page `FINDINGS.md`: what won on each axis, the dominant failure modes with gallery links, and recommended v2 changes.

## 18. v2 candidates (do not build)

Mug, bottle, pan (orientation-sensitive objects) · fine-tune an open VLM (Tinker) on sim-generated GT marks (marks for GT grasps are free to generate) · candidate-marker selection and iterative refinement · render-and-verify loop · view-weighted conflict policy and re-query · nearby obstacles in plots · front (y/z) view · few-shot examples · "around it, don't close" and "grasp the rim/body" prompts · more waypoints · integration as a `GraspProposer` in ChefBot.
