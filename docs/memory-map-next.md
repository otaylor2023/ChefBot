# Next steps: the kitchen memory map (physical robot, and offline training)

Written 2026-11-05 for a fresh Claude Code session. Goal: a **cheap memory map of the kitchen**: from the robot's
images, what each object is and roughly where it sits (which counter, x/y within a few cm), kept up to date as
objects move, so the base can drive to "spatula 1". Precision isn't the point: at the stance the arm re-finds the
object with the wrist camera. Read [kitchen-runbook.md](kitchen-runbook.md) for running the robot.

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
   comment). The old (11-03) `T` in the repo is wrong for these passes by ~12° (mostly roll): ~10 cm at 1.5 m.
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
