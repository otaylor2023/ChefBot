# Next steps: the kitchen memory map (physical robot, and offline training)

Written 2026-11-05 for a fresh Claude Code session. Goal: a **cheap memory map of the kitchen**: from the robot's
images, what each object is and roughly where it sits (which counter, x/y within a few cm), kept up to date as
objects move, so the base can drive to "spatula 1". Precision isn't the point: at the stance the arm re-finds the
object with the wrist camera. Read [kitchen-runbook.md](kitchen-runbook.md) for running the robot.

## Status 2026-11-06: the offline pipeline exists (branch memory-map, chefbot/perception/)
Gap 1 measured (fit, not marker-PnP), gap 3 done (YOLOE), gap 4 done (map + goto); gap 2 has a fallback. Run (vision env
for step 1; anything with numpy/PIL/yaml for the rest; batcomputer paths under /mnt/d/.../Engineering_AI_Agents):

    # 1. detect: YOLOE-26 text prompts (GPU: --device 0 on batcomputer's RX 6800, mps on the Mac)
    vision_venv/bin/python chefbot/perception/detect_frames.py STREAM/p2_layout3 --model yoloe-26s-seg.pt \
        --classes chefbot/perception/vocab_kitchen.txt --device 0 --out DETS --weights-dir models/yoloe
    # 2. map (and update it with a later pass: --init the earlier map)
    python -m chefbot.perception.memory_map build STREAM/p2_layout3 --dets DETS --config CAL.yaml \
        --spec maps/src_kitchen/spec.yaml --out maps/src_kitchen/objects.json
    # 3. drive to it (robot; --plan-only first)
    python -m chefbot.scripts.real_nav goto tomato --map maps/src_kitchen --hold --clearance 0.7 --turn-in-room
    # checks: chain_check.py (camera chain vs the map), memory_map probe (where a point falls in a frame and what
    # depth hits there), export_yolo.py (pseudo-labeled training folders)

CAL.yaml = the 11-05 pole calibration (`calibration/environments_real_config_2026-11-05.yaml`); its fitted mount offset is
`chefbot/perception/pole_mount_fit.yaml` (used automatically for that T only). Results: docs/current-state.md 2026-11-06.
Open: Jack's object list (prune vocab_kitchen.txt) and what moved in pass 3, to score the map; marker-PnP cross-check;
live updates in collect_stream (MemoryMap.update per frame); the deep-target rule with the teammate.

## Read first: the gaps between this and a working "go to tomato" (checked 2026-11-05)
Driving to a point beside a counter is solved and proven (`real_nav park --target X Y --counter C`, 200+ kitchen
trips). The image → world → stance chain is not. In order of how badly each would fail:

1. **Base frame mismatch, ~18 cm along the heading (silent).** The pole calibration's `T` is in the arm base's x/y
   (`config/robot_tidybot_fr3.yaml` `arm_mount: xyz [0, 0, 0.335]` = arm directly above "the base"). Our robot pose
   (mocap, `Localizer.base_from_mocap`) is the base's turning centre, which `interface/workspace.py` `Body.base_xy`
   puts ~18 cm **behind** the arm (itself ASSUMED, not measured). Chaining them as-is shifts every object ~18 cm along
   the robot's heading, plus the +0.105 m height offset. **Fix offline, no arm needed:** the counters' Motive markers are
   bright dots in the pole's IR images (`ir1/`, `ir2/`) at known world positions (`motive_descriptions_*.log` give
   the bodies' marker offsets; the live positions are in `mocap_markers_push_20261105.log` and in this doc's
   predecessors: island corners (0.17, −2.51), (0.22, −1.55), (2.43, −1.67), (2.39, −2.62), z 0.91). Find them in a
   few frames, solve the camera's world pose (PnP), and compare with `planar(base) @ T`: the difference is the
   arm-base ↔ turning-centre offset (and any calibration error). Validate the whole chain here before anything else.
2. **Many object positions have no stance.** `robot/nav/goal.py` `counter_park_candidates` only parks for targets
   ≤ `max_inset` 0.25 m in from a counter edge and ≥ 0.35 m (`PARK_END_MARGIN_M`) from its ends: the pick skill's
   trained start (inset 0.10–0.16, `TRAINED_INSET_M`). The island is 0.96 m deep, so an object in its middle gets no
   stance. Decide with Jack and the teammate: allow deeper targets (outside the pick's training), or approach from the
   nearer side and let the arm reach.
3. **No vision model ready to run.** batcomputer has no vision env (AMD GPU: needs ROCm PyTorch + transformers for
   `runtime/vision_server.py`); Gemini (`runtime/models.py`) needs an API key; the Mac (M1, MPS) can run OWLv2 slowly.
   Nobody has run a detector on these images: whether "tomato" / "spatula" are found from the pole at 1–2 m, partly
   behind the locked arm, is the first thing to measure on a few frames.
4. **No glue yet:** no memory-map file format, no builder, no `go to <name>` that looks an entry up, picks its counter
   and side (which `spec.yaml` box contains its x/y) and calls `park`. Small, but missing.

Smaller:
- The repo's `environments/real/config.yaml` still has the 11-03 pole `T`: anything live (`RealEnv.camera_pose`) is ~12° off.
- The wrist camera's pose uses the same 0.335 m mount (0.105 m low) and the arm's last joint readings (arm locked).
- The pickup can't be tested: the arm stays locked until its setup is reviewed (it sagged once). "Success" for the
  base: the object is in front of the parked base within the arm's reach, and in the wrist camera's view at READY.
- Needed from Jack: **the list of objects** on the counters (detector queries and the answer key) and **what moved in
  pass 3** (`MOVES.txt`).

### Shortest path to "go to tomato"
1. Offline: verify the camera chain against the Motive markers in the IR images (fixes gap 1; update the
   arm-base offset where it belongs, with the teammate).
2. Offline: one detector working on batcomputer (or the Mac); score it on ~20 frames per layout against Jack's list.
3. Offline: memory-map JSON + builder from a pass + `go to <name>` (a `real_nav` subcommand or a small script), with
   the deep-target rule from gap 2.
4. Robot, ~15 min: one recorded pass, build the map, `go to` three named objects; for each, check in the pole and
   wrist images that the object sits in front of the parked base.

## Where everything is
| what | where |
|---|---|
| data (3 passes, 971 MB) | batcomputer `/mnt/d/Jack/Documents/Stanford/Course_Work/Engineering_AI_Agents/chefbot_data/2026-11-05/` (README.md there); Mac `Claude_Code/robot_data/` |
| code | branch `kitchen-nav` (PR #2, stacked on PR #1 `movement`); batcomputer `.../Engineering_AI_Agents/ChefBot_fresh` is checked out there; venv `.../chefbot_venv` |
| kitchen map | `maps/src_kitchen/` (`spec.yaml` boxes + heights, `heading_zones.yaml`, `places.yaml`) |
| new pole calibration | `chefbot_data/2026-11-05/calibration/environments_real_config_2026-11-05.yaml` (`cameras.pole.T`) and `pole_card_2026-11-05.json` (the teammate's, 2026-11-05 15:51) |
| batcomputer rules | everything under `/mnt/d/...`, never `~` (small root disk); ask before long jobs |

## The data (per pass: `stream/<pass>/`)
| pass | layout | trips | notes |
|---|---|---|---|
| p1_layout2 | 2 | 20 | |
| p2_layout3 | 3 | 20 | |
| p3_moved_unseen | 3, Jack moved objects during the pass while the base faced away | 30 | pole USB dropped 14x (67 pole frames vs 143 wrist). What moved: ask Jack, then write `MOVES.txt` |

- `pole/`, `wrist/`: `color/`, `depth/`, `ir1/` (`ir2/` pole) PNGs, 1 frame/s while the base moved, same `n` across
  streams of one instant. `frames.jsonl`: `t_color_ms` / `t_depth_ms` (host clock, ms), `wall`, `blur`, `depth_valid`,
  `exposure_us`. `meta.json`: intrinsics + distortion for color / depth / IR, `depth_to_color` and `ir_from_depth`
  extrinsics, `depth_scale_m` (0.001).
- `state.jsonl`: changes at ~200 Hz on the same host clock (`wall`, s): `chefbot-tidy::pos` `[x,y,z]` and `::ori`
  `[qx,qy,qz,qw]` (Motive, z up, m), every mapped body, `hb1::current_pose` (odometry), arm joints (locked arm).
- `park_sessions/<run>/<nn>_<side>/park.json`: each trip's target and stance; `leg*.jsonl`: the drive at 20 Hz.

## From a pixel to the kitchen (the chain to implement)
1. **Time:** match each frame's `t_color_ms / 1000` to `state.jsonl` by `wall`; interpolate the robot pose.
2. **Robot base in the world** from the mocap body (`onboard/frames.py` `Localizer.base_from_mocap`, values from
   `onboard/config.yaml`): `yaw = yaw_of(ori) − 3.14°`; `base_xy = pos_xy − R(yaw)·(0.072, −0.002)`.
3. **Camera in the base:** `T` = the NEW `cameras.pole.T` (OpenCV axes: x right, y down, z forward). Its z is relative
   to the sim's arm mount (0.335 m); the real plate is 0.44 m up, so add **+0.105 m** to camera heights (the config's
   comment). Its x/y are the **arm base's**, not the turning centre our pose uses: ~18 cm apart along the heading
   (gap 1 above; measure it before trusting any position). The old (11-03) `T` in the repo is wrong for these passes
   by ~12° (mostly roll): ~10 cm at 1.5 m.
4. **Depth to 3D:** depth PNG × `depth_scale_m` → metres in the **depth** camera; deproject with the depth intrinsics,
   move to the color camera with `depth_to_color` (pole color is 1280x800, depth 1280x720: they are not aligned).
5. **World:** `p_world = planar(base x, y, yaw) @ T_pole @ p_color_cam`.
6. **Check it:** counter-top points should come out at the Motive heights (test counter 0.93, island 0.91, stove
   counter 0.83 m) and inside the counter boxes of `maps/src_kitchen/spec.yaml`.
The wrist camera's pose needs arm FK (`interface/workspace.py`, joints in `state.jsonl`; the arm was locked, last
readings); use the pole first.

## Offline pipeline (cheap first)
1. **Keyframes:** the 1 Hz frames, minus blurry ones (`blur` low) and frames whose pole view is mostly the arm.
2. **What is it:** an open-vocabulary detector on the pole color frames: OWLv2 or Grounding-DINO
   (`runtime/vision_server.py` already wraps both) with the object list as queries. Get the list of objects from Jack.
   A VLM call per keyframe (Gemini, `runtime/models.py`) is the fallback for labels the detector misses. Budget: a few
   hundred frames per layout.
3. **Where is it:** median depth inside the box (or its SAM 2 mask) → world point (chain above); keep points on a
   counter top (inside a counter box, height within ~5 cm of its top).
4. **Memory entries:** cluster detections per label in x/y (~5 cm): `{name ("spatula 1"), label, counter, xy, z,
   n_views, first_seen, last_seen, frames}`. Name duplicates left to right along each counter (agree the rule with Jack).
5. **Keeping it current:** each new frame either re-sees an entry (update it), or looks at the entry's spot (inside
   the image, depth there consistent with the counter) and doesn't see it → evidence it moved; enough of that → mark it
   gone. New detections far from entries → new objects. `p3_moved_unseen` + `MOVES.txt` is the test.
6. **Drive to it:** an entry's xy is a park target: `real_nav park --target X Y --counter <its counter>` (the stance rule
   in `robot/nav/goal.py` `counter_park_candidates`). That closes "go to spatula 1".
7. **Score:** per layout, Jack's object list as truth: found / missed / wrong label, position error (pole), and for
   pass 3 whether moved objects were caught.
Training a detector is only worth it if the pretrained ones miss the kitchen's objects; measure first.

## Physical robot next steps
1. **Calibration into the code:** take the teammate's new `cameras.pole.T` into the branch (agree with them: their
   `~/ChefBot` copy is uncommitted). Re-check later: the pole mount "isn't very stable" (their note).
2. **USB:** pole D455 straight into a computer port (stops the drops through the shared hub; may bring back its IMU);
   a working USB 3 cable for the wrist D405 (1280x720 instead of 640x480).
3. **Network:** Mac on SRC, or keep driving from the robot (runbook section 3).
4. **More layouts, small:** one 20-trip pass per layout at 1 frame/s; for change tests, write down what moved and
   when (MOVES.txt).
5. **Map gaps:** measure the west/north walls and L leg 2's depth; add markers at the corners still missing.
6. **Speed:** planning a 180° trip across the island takes ~6 s (pure-Python A* repairs); the south→north path zig-zags.
7. **Arm (blocked):** an experienced operator reviews the arm's setup (it sagged when unlocked) before any arm-on run;
   then the READY pose at each stance gives the wrist camera's view for re-identification.
8. **Merge:** PR #1 then PR #2 with the teammate; the 0.5 m/s cap (branch `speed-exp`) is their decision.
