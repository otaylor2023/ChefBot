# ChefBot current state (verified)

Last updated: 2026-11-03 by Claude Code (read-only inspection, nothing launched, no Redis writes).

## Access
- Mac → robot: SSH alias `chefbot` = `tidybot01@100.124.41.87` (Tailscale). Password auth, no key.
  Jack opens a shared connection: `ssh -fN -o ControlMaster=yes -o ControlPath=~/.ssh/cm-chefbot -o ControlPersist=8h chefbot`.

## Robot (tidybot01-GTi), observed 2026-11-03 ~12:00
- Booted 2026-11-03 11:41. Wi-Fi `wlp88s0` on **SRC**, 172.24.69.172/23 (not the 10.36.x of the handoff).
  `enp90s0` 172.16.0.69/24 (arm network). Tailscale 100.124.41.87.
- Running: only `redis-server 127.0.0.1:6379`. No SampleClient, base driver, OpenSai, or python apps.
- Redis persists to disk (`save 3600 1 300 100 60 10000`, dir /var/lib/redis) → **keys survive reboots**.
  Present now (stale, publisher not running): `chefbot-tidy::pos=[0.394562,0.40328,0.443938]`,
  `chefbot-tidy::ori=[0.00435,-0.00383,0.25715,0.96635]` (xyzw), `chefbot-tidy::tracking_valid=1`;
  also `tidybot01::{pos,ori,tracking_valid}` and `hb1::{current_pose,current_vel,desired_pose,desired_vel,stop,kill}`.

## ~/ChefBot on robot: NOT a git checkout
Partial copy (no `.git`), files dated 2026-11-02 15:52–18:23:
`chefbot/onboard/{__init__,keys}.py, config.yaml, bringup/{launch.sh,mocap.sh,README.md,chefbot_fr3.xml},
calibration/{arm,camera,fr3,load,wrist}.py`, plus `~/ChefBot/Client-output.pts` (2026-11-02 17:23).
**Missing on robot:** `server.py`, `robot_io.py`, `frames.py`, `check.py`, `cameras.py`, `pyproject`; no `chefbot`
conda env (envs: clean-env, dreambc_robot, tidybot2, zitibot). So `launch.sh` step 5 and `python -m chefbot.onboard.check` cannot run as deployed.
Exact source commit of the deployed copy: **unknown**.

## Mocap path (as deployed)
- `mocap.sh`: requires wlp88s0 IP 172.24.*; MOTIVE_IP 172.24.69.102; MOCAP_MODE unicast (default).
  Builds once: sed-copies SDK `SampleClient.cpp` → `~/chefbot_mocap/SampleClient_unicast.cpp`, only change is
  `kDefaultConnectionType = ConnectionType_Unicast` (diff verified). Binary `~/chefbot_mocap/build/SampleClient_unicast`
  built 2026-11-02 17:21. Runs `SampleClient_unicast 172.24.69.102 <my SRC IP>`; writes `Client-output.pts` in the cwd.
  Only writes Redis (no drivers started).
- Client publishes per rigid body every frame: `<asset>::pos` [x,y,z], `<asset>::ori` [qx,qy,qz,qw],
  `<asset>::tracking_valid` "0"/"1" (params&0x01). pos/ori are written **even when tracking_valid=0**.
  **No frame ID or timestamp key** → freshness cannot be read from Redis directly.
- ChefBot reads `chefbot-tidy::*` (keys.py). config.yaml: `marker_yaw_offset_deg: 0.0`, `marker_xy: [0,0]` (TODO measure).
- `launch.sh` DELs only `chefbot-tidy::tracking_valid` before starting mocap, waits ≤10 s, then continues
  without it. `mocap.sh` alone does not clear anything.
- Evidence of earlier reception: `Client-output.pts` (2026-11-02 17:23) contains a marker data block with frame
  numbers 686421…687023 — someone received ~600 frames from Motive on Nov 2. Not verified today.

## Consumer (from ~/Downloads/ChefBot-main.zip, dated 2026-10-28 — older than robot copy, may be outdated)
- `robot_io.read_state`: if tracking_valid==1 and pos/ori/odom parse → `frames.tie(base_from_mocap(...), odom)`.
  **No freshness check.** A dead publisher leaves tracking_valid=1 → every state read re-ties odometry to the
  frozen pose, so the reported world pose stays frozen while the base moves, and world→odom targets are wrong.
- `frames.yaw_of_quat_xyzw` assumes z-up. Stale ori above ≈ 30° about z, consistent with a z-up Motive frame
  (inference, not verified).
- `launch.sh` base limits 0.125 m/s, 0.4 rad/s; server watchdog 0.3 s (client silence).

## Open items / unverified
- Where the current full ChefBot repo lives (teammate's GitHub? not found on Mac; `gh` not installed).
- Live frames today; asset name `chefbot-tidy` streaming in Motive; Motive transmission = unicast.
- Mocap freshness/dropout handling in the current server code.
- Frame alignment (yaw offset, marker_xy). Arm sag issue unresolved — arm must not be unlocked.

## Next action
Mocap-only test (no motion): run `mocap.sh` in the foreground ~10 s, confirm FrameID increments, `chefbot-tidy`
listed with Tracked=1, Redis pos/ori change from the stale values above.

## Legacy ZitiBot navigation (read 2026-11-03; ZitiBot git HEAD 7530046 "full working scrambled eggs!", 2026-06-09)
- Stack: `tidybot_base/redis_driver.py` (hb1 driver, wheel odometry, tracks `hb1::desired_pose`, `hb1::stop`,
  live `hb1::max_vel_scale` clamped [0.1,3.0]) + Motive publisher started manually (`tidybot01::*`) +
  a controller that calls `zitibot_core/base.py: go_to_pose(ctx, BaseWaypoint.X | x_m,y_m,yaw_deg, motion=holonomic|three_phase)`.
- Engine: `tidybot_base/opti_nav.py: navigate_to_opti_pose / run_replan_loop` at 100 Hz: map Motive goal into
  odometry (`opti_planner.MocapToHbCalib`, p_hb = R p_mocap + t), write `hb1::desired_pose`, replan from live mocap,
  stop+wait if `tracking_valid` false, stall exit 4 s, divergence warning 4 s, tol 1 in / 5°.
  Rotation seeded by `DEFAULT_MARKER_YAW_OFFSET_DEG = -38.17` (comment says +140.14, archive doc says +41.0),
  then refined online by `MotionDirectionEstimator` after ≥8 cm motion.
- Stations: `zitibot_core/constants.py BASE_WAYPOINTS` (Motive x,y,yaw; e.g. STIRRING (0.8555,-4.0656,-90°)), taught
  for the old `tidybot01` rigid body.
- Tools: `tune_marker_offset.py` (20 cm hb +X move → offset; has `--mocap-*-key` flags), `test_base_axes.py`
  (small hb x/y probes vs prediction), `station_nav_test_controller.py` (station-to-station, logs target vs actual).
- Known gaps: stale-publisher guard was removed (relies on tracking_valid only; a dead publisher leaves it =1);
  driver has no command-freshness timeout; offset and waypoints belong to the old asset, not `chefbot-tidy`.

## 2026-11-03 12:22 — teammate active on robot; repo location
- No full ChefBot git repo anywhere on the robot disk. Teammate copies files in from their machine:
  12:02–12:20 today `calibration/pole.py`, `config/robot_tidybot_fr3.yaml`, `bringup/{launch.sh,arm.sh}` (new
  `--arm-driver` flag), `calibration/{arm,wrist}.py`, `chefbot/environments/real/config.yaml` (envs/ → environments/ rename?).
- Teammate SSH from 100.76.239.18 since 11:58. Running since 12:07: `sai_franka_robot_redis_driver` (arm torque
  driver via safe_launch_driver.sh), `OpenSai_main chefbot_fr3.xml`, web UI (tmux interfaces_server). Base, gripper, mocap off.
- `~/chefbot_calib/`: arm load calibration (`load.json`, `load_check.json`, 2026-11-02), `arm_log.jsonl`, wrist/pole images and JSON today.
- Full repo snapshot available locally: `~/Downloads/ChefBot-main.zip`, commit 3be292f (2026-10-28) — outdated.

## 2026-11-03 12:50 — repo and movement branch
- Real repo: github.com/otaylor2023/ChefBot. Local clone `Claude_Code/mac_chefbot`, main = 3be292f (= GitHub main; nothing newer pushed).
  Other remote branches: explore-map (Simone, 10-28), feat/recipe-planning (Jake, 10-17).
- Robot copy is ahead of main: 17 of 19 files new/changed, unpushed.
- Worktree `Claude_Code/mac_chefbot-movement`, branch `movement` (local only, not pushed):
  - 54a0ca4 snapshot of robot `chefbot/onboard` at 2026-11-03 12:45 PST (teammate's work, verbatim; environments/ and config/robot_tidybot_fr3.yaml left out).
  - 9a6ceca `chefbot/onboard/nav/README.md`: Jack's section plan, no code.
- No local Python env with pytest yet (homebrew python3 lacks pytest/numpy); tests not run.

## 2026-11-03 ~13:10 — stage 0 code
- Branch `movement`: removed the onboard/nav commit (nav is laptop-side logic). Now:
  3be292f (main) → 54a0ca4 (robot onboard snapshot) → 85c2ef2 (stage 0).
- 85c2ef2: onboard/mocap.py (freshness), robot_io tie only valid+fresh, new state fields, server halts on
  disconnect only after commands, RealEnv.localization(), scripts/real_pose.py. 297 passed, 1 skipped.
- Found: server.py on main wrote arm goal + controller switch + hb1::stop on every disconnect/shutdown, even for a watch-only client (fixed in 85c2ef2; needs teammate review).
- Found: `httpx` imported by agent/loop/llm.py but missing from pyproject.toml.
- Local venv: `Claude_Code/.venv-chefbot` (Python 3.11, `pip install -e mac_chefbot-movement[robot,dev]` + httpx).
- Robot: no env with websockets (teammate's ~/chefbot_calib/env lacks it). HT-0a planned in docs/hardware-tests.md; not run.

## 2026-11-03 13:07 — HT-0a PASSED (stage 0, live mocap on laptop, no motion)
- Robot: `~/chefbot_movement` = 85c2ef2, conda env `chefbot-movement`. Both left in place; server and mocap stopped after the test.
- 100% live while publisher up, jitter std ≈ 0.02 mm / 0.003°, stale detected 0.265 s after publisher stop, no command keys written.
- Blocking findings: stale `hb1::current_pose` (needs freshness), Motive up axis likely Y (check Streaming → Up Axis). Details in docs/hardware-tests.md.

## 2026-11-03 — stage 1 tools ready (not run)
- 6c0d9db + da679df on movement, deployed (DEPLOYED_COMMIT da679df): onboard/calibration/base.py (record/solve: up axis, offset per labelled move, mirror check, marker_xy) and onboard/bringup/gamepad.py (tidybot2 teleop at ≤0.2 m/s, 0.5 rad/s). 306 passed, 1 skipped.
- Jack's marker description (all on one plane, two slightly raised) matches markers #5/#7 at body y=+0.01 → body y is up; body ≈ aligned with world → Motive streams Y-up (to confirm with HT-1a).
- tidybot2's gamepad_teleop.py hard-codes 1.0 m/s / 3.14 rad/s at full stick.

## 2026-11-03 ~13:40 — correction: Motive streams Z-up
- Retracted the Y-up inference (it came from the rigid-body description offsets). Frame data: base turned 9.2° about z between HT-0a and now; z height constant ~0.44 m. frames.py's z-up yaw is correct; no frames.py change needed.
- G0 (stick echo) and G1 (powered direction check) passed per Jack. Swerve modules lurch while re-steering at a direction change → solver now fits the steady part (d1dc224, deployed).
- mocap.sh running in background on robot (PID 11279) for HT-1a.

## 2026-11-03 13:45 — HT-1a PASSED (stage 1: mocap frame from driving by hand)
- Measured: up z; marker_yaw_offset_deg 3.14 (4 moves agree within 0.6°); marker_xy [0.072, -0.002] (std 1 mm). Committed ee7cb88 (onboard/config.yaml), deployed to ~/chefbot_movement. Teammate's ~/ChefBot config still 0.0 / [0,0].
- Validation: base centre 1.4 mm rms through a 379° spin; heading vs travel within 0.55°.
- Mocap stopped. Gamepad and base driver not running.
- Next: stage 2 (base driver copy with command timeout + odometry freshness), then stage 3 bounded moves.

## 2026-11-03 — stages 2-3 code ready (not run on hardware)
- 892edae + ad47a92 on movement, deployed: onboard/drivers/base (ZitiBot Vehicle copied verbatim + driver.py with 0.5 s command timeout, heartbeat), robot_io driver liveness + NotLocalized guard, bringup/base.sh, scripts/real_nav.py (mark/point/list/go). 327 passed, 1 skipped.
- Robot non-moving checks passed. Teammate's arm stack still running → base.sh refuses until stopped.
- Procedure HT-2a/HT-3a in docs/hardware-tests.md.

## 2026-11-03 14:00 — FIRST AUTONOMOUS MOVES (HT-3a) PASSED
- A→S1 0.3 m: 4.2 mm / 0.47°; S1→A: 6.7 mm / 0.06°; both ≤ 3 mm off the straight line. Through laptop real_nav → server → ChefBot driver, mocap closing the loop.
- HT-2a: driver timeout fired with server frozen (log line); stop distance not measured → re-run.
- base.sh exited after the test (server killed by design); needs restart.
- 14:22 characterization suite ready (a97e144, deployed); HT-3b planned in hardware-tests.md

## 2026-11-03 14:35 — HT-3b PASSED; robot handed to the teammate
- 52/52 legs reached in a ±0.2 m fence: repeatability 0.2 mm std; final errors ≤ 8.3 mm, yaw ≤ 0.82°.
- Found + fixed: base.sh could not be stopped (dad29a2, deployed, tested with stand-ins). Open: per-axis speed limit (diagonals up to 0.167 m/s measured); laptop report's odometry-scale metric meaningless; turn centre drift up to 15 mm.
- All our robot processes stopped; port 8765 free; tunnel closed. Teammate using the robot now.

## 2026-11-03 ~15:00 — offline while the teammate has the robot
- ca9695f: driver planar speed cap along the travel direction (fixes 0.167 m/s diagonals); real_nav peak speed from server times; meaningless odometry ratio removed.
- 9b17cc2: scripts/real_map.py (map spec → grid + places), real_nav go --map (A* around obstacles). Footprint 0.33 m from Jack's "≤ 10 in past the mocap body"; margin 0.10 m for the arm at the front.
- Not deployed to the robot yet (teammate using it). 337 passed, 1 skipped.
- Kitchen priors: ZitiBot stations in the same Motive frame: counter on +y side, stations y ≈ −2.8 (x 0.34–1.6) facing +y; counter on −y side, stations y ≈ −4.07 (x 0.82–1.5) facing −y; oven door (0.88, −3.93). Old asset's marker positions: rough only.
- A* test needs ~3 x 3 m clear floor (0.43 m keep-out each side of obstacles and bounds).

## 2026-11-03 ~15:20 — counter-top map; teammate running the full stack
- 6f497ef: random A* trials (scripts/real_trials.py) with imaginary obstacles + oriented footprint (base box + arm boxes), onboard/sim.py simulator (runs as its own process). Sim: 30/30 reached, 16 detours, 0 violations.
- 4a5da96 (deployed to ~/chefbot_movement): maps/src_counter from Motive rigid body `Chef-botcounter-top` (ID 25): box centre (1.49, 2.06), 2.26 x 0.96 m, yaw 0, markers 0.93 m up. Bounds PROVISIONAL. Sim lap of the counter: 5/5 reached, base centre ≥ 0.458 m from the counter (needs 0.43).
- Motive rigid-body descriptions are in y-up; the stream is z-up. Mirror sign unresolved; the counter box covers both.
- 15:20 robot: teammate running arm driver, gripper driver, ZitiBot base redis_driver.py (no command timeout), mocap, and an onboard server from ~/chefbot_calib/env. Not touched.
- Needed from Jack: the room's free-floor bounds (walls, cabinets, Motive table) in mocap coordinates.

## 2026-11-03 15:40 — arm in the footprint (6303d77, laptop-side; robot still at 4a5da96)
- robot/kinematics/arm_outline.py: arm over the floor from joints (Franka DH; ready-pose flange matches within 2 mm). Arm mount (0, 0, 0.335) assumed from teammate's config.
- robot_io: arm_live. real_nav: footprint covers the arm (--arm auto), `real_nav arm`, stop if a joint moves > 0.05 rad while driving. 348 passed.
- Read-only from Redis (not via teammate's server): arm now reaches 0.55 m ahead → footprint 0.56 m; stow posture 0.23 m → 0.38 m. Joint values not changing (driver not streaming).
- With a stowed arm only near_side and west_end plan; going around needs the real room bounds.
- Map page rewritten (v17): goals, stages, accomplishments, assumptions, problems, plan.

## 2026-11-03 15:53 — HT-4a PASSED: hugging the counter on the real robot
- 3/3 legs along the counter's near side (f5ca190): final errors 0.3 / 6.7 / 6.1 mm; base centre ≥ 0.713 m from the counter, arm box gap ≥ 0.31 m, 0 overlaps.
- base.sh stopped cleanly (fix confirmed). Nothing of ours running; tunnel closed. Teammate's stack was stopped for this test.

## 2026-11-03 16:02 — HT-4b PASSED: full lap around the counter at 0.15 m/s, facing each side
- 5/5 legs, 10.79 m, 696° turned; base centre ≥ 0.625 m, arm box ≥ 0.081 m from the counter, no overlaps.
- Lurch bursts up to 0.229 m/s seen by mocap (wheels ≤ 0.157): keep margins ≥ 5 cm at this speed.
- Everything stopped cleanly; tunnel closed; robot free.

## 2026-11-03 16:10 — optimization plan (Jack's HT-4b observations)
1 oriented footprint in planning (base box + arm box at heading) · 2 far-left wall + closer lap · 3 HT-2a stop distance
(before faster) · 4 smooth path following · 5 speed 0.2–0.25 with margins from 3 · 6 arm mount (tape now; touch-off with
teammate later, guiding mode = the mode of the earlier drop) · 7 live counter pose in the server + a kitchen profile file.
Open question to Jack: where is the far-left wall?

## 2026-11-03 ~16:30 — robot geometry from Jack; counters vs walls (3bc7bb8, c30cf93 deployed)
- Jack: the base is 44 cm tall, the arm on top → ARM_MOUNT z 0.44 (was Menagerie's 0.335). Locked arm tip 1.047 m,
  11.7 cm over the 0.93 m counter-top markers (matches Jack's view). The teammate's pole-camera height uses 0.335 too.
- Walls (Jack, roughly): east wall 53 in past the counter's east end (x ≈ 3.97), far wall 6 ft behind the far side (y ≈ 4.37);
  west end and near side open. In maps/src_counter (5 cm closer each).
- Layered planning: counters are low (arm may pass over if ≥ 5 cm above; per-point thickness), walls tall.
  Locked arm: 0.55 m vs walls, 0.33 m vs the counter. Close lap (0.45 m out) plans at 9.39 m.
- No robot description holds base height / arm mount: belongs in one robot + kitchen profile.
- 16:30 teammate running full stack again (incl. ZitiBot base driver). Not touched.

## 2026-11-03 17:01 — HT-4c PASSED: stop distance measured; close lap with the base ~10 cm from the counter
- Stop: 0.119 m/s → rest in 1.11 s / 66 mm after commands stopped. Worst case at 0.15 m/s ≈ 19 cm → stronger timeout braking next.
- Close lap 6/6, ≤ 3.5 mm; base body ≥ 9.6 cm from the counter, low arm parts ≥ 9.9 cm. Lurch bursts to 0.30 m/s (mocap).
- Robot rebooted 16:40 (back on SRC). All stopped; tunnel closed.

## 2026-11-03 ~17:30 — braking, smooth following, square stations (deployed)
- Driver: brake 0.25 m/s² / 0.8 rad/s² (gamepad's proven limits), timeout 0.3 s → worst-case stop ~9 cm at 0.15 m/s (was ~19). Not yet measured on hardware.
- Smooth following (BaseController carrot 0.12 m; 0.3 m cut corners past the margin in a dry run) and square stations (counter_*: 0.45 m out, 0.25 m straight approach). Sim station lap 5/5, ≤ 22 mm off path.

## 2026-11-03 17:21 — HT-4d PASSED: braking halved, smooth station lap, square finishes
- Stop: 78 mm / 0.90 s from 0.131 m/s (was ~16 cm predicted at this speed with the old settings). Worst case at 0.15 ≈ 10.5 cm.
- Station lap with smooth following: square within 0.54°, 4 mm; base body ≥ 6.6 cm from the counter.
- One Wi-Fi stall (~0.3 s) tripped the server watchdog → held → retried. Next: run the motion loop on the robot (or wired).
- All stopped; tunnel closed; robot free.

## 2026-11-03 ~17:30 — robot data pulled to the Mac
- `Claude_Code/robot_pull/20261103/` (74M): `chefbot_movement` (deployed a9919f5 + raw `runs/` ht0a–ht4d, bringup_base),
  teammate's `ChefBot` onboard copy, `chefbot_calib` (no env/), `chefbot_mocap`. 413 files, sha256 robot == local (MANIFEST.txt).
- Deployed source identical to git a9919f5; local `movement` branch is one commit ahead (6d1cc18), still unpushed.

## 2026-11-03 ~18:00 — movement rebased onto main and pushed
- GitHub main moved 3be292f → 8bdd32d (99 teammate commits: envs/→environments/, config/arms→embodiments/, real host, calibration, float.sh).
  Teammate's robot `~/ChefBot` = main for 48/52 files (workspace.py, environments/real/config.yaml, calibration/arm.py differ; uncommitted on robot).
- Old history tagged `archive/movement-2026-11-03` (= 6d1cc18), pushed. All hardware-test hashes stay reachable.
- `movement` rebased onto 8bdd32d, snapshot 54a0ca4 dropped. Conflicts resolved (kept both sides): server.py (cameras idle + halt only if commanded),
  robot_io.py (arm_frozen + driver_live odometry), environments/real/env.py (state_age + localization), test_onboard.py.
  + fd88dd4 import fix (envs.real → environments.real), + docs commit (docs/hardware-tests.md, docs/evidence/). Head e41da10, pushed to origin/movement.
- Tests (Mac venv, mujoco 3.17.0): 365 passed, 8 failed, 2 skipped (ignoring tests/autoresearch: needs imageio/PIL). The 8 failures are in
  test_real_host.py (MjData has no qM) and fail identically on unmodified main.
- Local `mac_chefbot` main fast-forwarded to 8bdd32d.
- PR: not opened yet (gh not installed). Body drafted. Robot unchanged: ~/chefbot_movement still a9919f5 (pre-rebase).

## 2026-11-04 ~15:40 — teammate review folded in; PR #1 still OPEN (not merged)
- GitHub main 0a3c7d7 (2699823: vendored ZitiBot redis_driver.py as onboard/drivers/base/driver.py, launch.sh starts it;
  0a3c7d7: body.plate_height_m 0.44, driver limits incl. accel in onboard/config.yaml, launch.sh reads them).
- Teammate review (pasted by Jack): rebase, keep Jack's driver.py/README. Requested: nav loop into Robot.go_to_pose/robot/safety;
  footprint from workspace.arm_points + body (arm ~18 cm ahead of base centre; plate corners ~0.37 m; HT-4d's 6.6 cm may be ~3 cm
  at a corner: verified body.plate.x [-0.462, 0.097]); speeds from speed_profiles; maps: build_map in environments/real, heights.npy,
  semantic_map.yaml; mocap as its own reading. Suggestions: CAN-thread liveness, 0.298 m/s lurch in HT-4c, key-drift test, arm-on run.
- Local `movement` rebased onto 0a3c7d7 (driver.py = PR version; README + redis_driver.py provenance), + 2 commits: driver exits
  without heartbeat when Vehicle.control_loop_thread dies, keys-match test; base.sh reads config limits. Head **74c082d**, 27 commits
  ahead of main. pytest: 367 passed, 8 failed (pre-existing test_real_host mujoco qM), 2 skipped.
- **Not pushed**: force-push of movement blocked by auto-mode permission; needs Jack. gh installed, not logged in.
- Robot: ~/chefbot_movement = 74c082d (git archive, 542 files sha256-verified); previous tree kept at ~/chefbot_movement_a9919f5
  (runs/ moved into the new tree). ~/ChefBot untouched.
- 15:24 someone (pts/0 from 100.76.239.18) started the FULL launch.sh from ~/ChefBot: arm driver, OpenSai chefbot_fr3.xml, gripper,
  vendored base driver (NO command timeout), mocap, server. Not touched. HT-5a..c wait until the robot is free.
- Counter lap paused until footprint fix (review C1).

## 2026-11-04 ~16:05 — HT-5a counter lap PASSED on 74c082d (+ pole images)
- base.sh from ~/chefbot_movement with CHEFBOT_PYTHON=~/chefbot_calib/env/bin/python (pyrealsense2). Arm off, locked.
- 5/5 stations (counter_south, east_end in place of counter_east, counter_north, counter_west, hug_a), 3–9 mm, ≤ 0.73°.
  Plate rectangle ≥ 5.7 cm from counter at margin 0.09. counter_east refused (approach point vs east wall, arm reach 0.548).
  Two link-stall holds (Mac 10.32.77.55 not on SRC; RTT ~88 ms) — retried. Peak speeds 0.146/0.174 vs 0.125 cap: open.
- Pole images: movement worktree runs/pole_snaps/20261104_lap/ (gitignored; people in frame). Depth valid 20–34%.
- Local movement = bc4af73 (74c082d + HT-5a docs). **Push still blocked by auto mode: Jack must run it.**
- Left running: base.sh on robot (Jack's terminal); Mac tunnel -L 8765 via cm-chefbot.

## 2026-11-04 ~16:30 — PR #1 pushed (bc4af73, fast-forward over main 0a3c7d7); nav performance inspection (no motion)
- PR #1 head = bc4af73 on GitHub; main is its ancestor (no conflicts possible). gh logged in (Jtop13).
- **Bug (safety): wrong-way base rotation across ±180° in the odometry frame.** frames.to_odom wraps the target yaw;
  robot_io.base_target sends it; the Vehicle's Ruckig treats yaw as linear → long way round. HT-5a east_end leg: ~285°
  wrong-way spin at ~15°/s near the counter's SE corner, 22 s lost (odom offset ≈ -58° this session). Not yet fixed.
- HT-5a speeds by driver odometry: max 0.098 m/s. robot_io scale = min(v/0.125, yaw_speed/0.4) = 0.75 with real_nav's
  --yaw-speed 0.3 → planar cap 0.094. HT-5a's 0.146/0.174 "peaks" are mocap-differencing artifacts.
- Per station: ~2 s of 1 s hold-tails (go() tail_s), a full stop at the approach point, 0.25 m straight-in ≈ 4 s, 0.5 s settle.

## 2026-11-04 ~16:55 — HT-5b done, robot handed to the teammate; pushed 5aae46a (PR #1)
- Robot ~/chefbot_movement = da85382 (heading fix; verified on hardware: 15 parks, every turn the short way).
- Laptop: 16797ec/fa9138d `real_nav park` (stances by the pick start rule; robot/nav/goal.py counter_park_candidates),
  5aae46a HT-5b docs. Direct park 8/8, −17% vs straight-in, 0 intermediate stops, ≤ 8.3 mm, ≤ 1.3°, plate ≥ 61 mm.
- Speed limiter found: carrot 0.12 m + Ruckig plan-to-stop → cruise ~0.072 m/s regardless of cap.
- Jack: park closer to the counter; move 2–3x faster or more without risking safety. Robot now used by the teammate.
- Unexpected stops: robot-side Wi-Fi dropouts (mocap and laptop link both on wlp88s0; −63 dBm).
- Tunnel closed. Nothing of ours running on the robot (teammate has it).

## 2026-11-04 ~17:10 — speed work committed locally (447c5c8), not pushed, not deployed
- Speeds from embodiments/speed_profiles.yaml (`--profile`, default robot control.speed = normal 0.2 m/s, 0.375 rad/s).
- robot/nav/speed.py: measured stop model (from the server's hello limits; onboard/config.yaml base_stop_tau_s 0.333,
  base_stop_brake_mps2 0.25 = HT-4d), clearance-limited speed along the direction of travel, speed-sized carrot with
  a clear-chord check. Park: still-and-in-tolerance 0.2 s, 0.3 s tail, stances 0.45 m (Jack: closer; arm clear of top).
- Driver sanity bounds 0.5 m/s / brake ≤ 1.0; real ceiling still config.yaml 0.125 m/s, 0.1 m/s².
- Tests 386 passed (8 pre-existing test_real_host mujoco failures). Sim park 8/8 at 0.45 m.
- Robot still da85382: park now REFUSES there (no measured stop in hello) until the robot side is redeployed.

## 2026-11-04 ~17:40 — fast config committed locally (5f7a53d + 394b24b), NOT deployed, NOT pushed; robot untouched
- onboard/config.yaml test ceiling 0.3 m/s / 0.8 rad/s / 0.3 m/s²; stop-model brake 0.3 is PREDICTED.
- Sim (honours scale, robot ceilings): park --random 8 --seed 1 --profile fast: 8/8, 85 s (HT-5b hardware direct 253 s).
- Robot still runs da85382 via Jack's base.sh. Jack can't test now.
- Next session: deploy → stop test at 0.3 m/s (SIGSTOP the server mid-move, calibration.base record on the robot) →
  put the measured tau/brake in config → park --random 8 --seed 1 --profile fast, Jack on e-stop.
- 2026-11-04 ~17:50: pushed movement 394b24b; PR #1 (movement→main) updated: title, body incl. the warning that
  5f7a53d raises launch.sh's driver ceiling to 0.3 m/s untested. GitHub: MERGEABLE, open, review pending.
- 2026-11-04 17:36: posted review-response comment on PR #1 (https://github.com/otaylor2023/ChefBot/pull/1#issuecomment-6647190609): per-request status with commits; open: nav loop into Robot.go_to_pose, footprint from body, map format, mocap reading in sims, arm-on run; decisions asked: 0.3 m/s ceiling before stop test, 0.45 m parking.

## Compute hosts
See [compute-hosts.md](compute-hosts.md): batcomputer set up at 80f7ad0 on /mnt/d (2026-11-05); Mac stays primary.

## 2026-11-05 ~11:30 — deployed 80f7ad0; HT-6a PASSED (stop at cap); 16/16 teleop-speed counter parks
- Robot booted ~11:16, SRC 172.24.69.172, nothing running. `~/chefbot_movement` = **80f7ad0** (sha256 == git, 571 files);
  old tree `~/chefbot_movement_da85382`; `runs/` moved across. `~/ChefBot` untouched (older than main fe2ed94 in onboard/).
- base.sh started by Claude (setsid, log `~/chefbot_movement/runs/base_sh_20261105.log`), CHEFBOT_PYTHON=chefbot_calib env.
  Mac tunnel -L 8765 on cm-chefbot.
- HT-6a: freeze server at 0.225 m/s → rest in 0.177 m / 1.2 s (model 0.176). Stop model confirmed; robot.yaml unchanged.
- HT-6e: `park --random 8 --seed 1|2 --profile teleop`: 8/8 + 8/8, 114 s / 112 s (HT-5b 253 s), ≤ 1.8 mm, ≤ 0.18°,
  0.45 m out, plate ≥ 92 mm, 0 stops. Details: mac_chefbot-movement docs/hardware-tests.md.
- Fixed in docs: HT-6a command must be `--profile teleop` (`--speed 0.25` over `normal` is refused).
- Open: HT-6b (untick chefbot-tidy in Motive mid-park), HT-6d map bounds (ymin −0.05 refused a start at y 0.19), HT-6c arm (blocked).

## 2026-11-05 ~11:50 — speed experiment (branch `speed-exp`, NOT main); pushed; robot handed back
- `speed-exp` = movement 3d63e84 + e42fe35 (cap 0.5 m/s in robot.yaml, accel/brake 0.25; profiles exp35/exp50) + evidence/docs
  (dd3f9d2, 67e33dd: docs/hardware-tests.md HT-7). Main's 0.25 cap is the teammate's; unchanged on main and on movement/PR #1.
- Stops (server frozen mid-leg): 0.323 m/s → 0.31 m; 0.478 m/s → 0.57–0.59 m (model 0.616). Model holds to 0.5.
- Parks seed 1 at 0.5 m/s: 8/8, 89.8 s (vs 114.4 at 0.25), but path deviation 118 mm and plate 33 mm from the counter
  (vs 55 / 92 mm). Open: tighter tracking at speed, higher accel/braking, teammate's call on main's cap.
- GitHub: movement 3d63e84 (PR #1 head, docs only on top of 80f7ad0), speed-exp 67e33dd (new branch, no PR). main fe2ed94.
- Robot: all our processes stopped (driver, mocap, server, recorders), port 8765 free, temp scripts removed, tunnel closed.
  Trees: `~/chefbot_movement` = 80f7ad0 (PR head code), `~/chefbot_speedexp` = e42fe35 (0.5 cap), old trees kept.
  `~/ChefBot` untouched (no files changed today). Base parked at the counter's west side.
- Local worktree mac_chefbot-movement back on `movement`.
- Lesson: background-started base.sh ignores SIGINT; stop it with `kill -TERM <base.sh pid>` (it traps TERM). Same for
  `calibration.base record` (TERM). Don't `pkill -f` a pattern that appears in the ssh command itself.

## 2026-11-05 ~12:30 — data collection while driving (branch `data-collect`, local, a81d389)
- Stop-and-shoot viewpoint grid (real_collect ring) dropped on Jack's call: not real motion, too slow. 19+ stops of it exist
  on the robot (`~/chefbot_collect/data/p1_ring`, ~150 MB), not part of the plan.
- Now: `onboard/collect_stream.py` on the robot (`~/chefbot_collect/`, run with chefbot_calib env) records both cameras at
  15 fps, every frame lossless, host-clock timestamps, plus robot state changes at 200 Hz, while the Mac runs
  `real_nav park --random 8 --seed N --profile teleop` (0.25 m/s). Start: setsid nohup ... & pid in /tmp/collect_stream.pid; stop: kill -TERM.
- s1_seed3: 8/8 parks, 74 s, 1180 frames/camera, 2.5 GB. s1_seed4: 8/8, 98 s, 1530 frames/camera, 3.1 GB. Objects as placed ~12:00.
  Pole sharpness driving ≈ still (no visible motion blur at 0.22 m/s). Disk 805 GB free.
- Wrist D405 stuck on USB 2.1 (480 Mbps) whatever port Jack tried; recorded at 640x480. Pole D455 USB 3.2 via a 10G hub.
- Base stack: base.sh from ~/chefbot_movement (80f7ad0) started 12:01 by Claude, still running; tunnel 8765 open.
- 12:30–12:55 more sessions (all 8/8 parks, teleop): s2_seed5 (3.6 GB), s2_seed6 (3.2 GB) on pass-2 objects; s3_seed7 (2.9 GB),
  s3_seed8 (pole stalled after 44 s: USB -71 errors; wrist full), s3_seed9 (2.9 GB, clean) on pass-3 objects (some clumped on purpose).
  INVALID_s2_seed5_objects_being_moved kept apart. Total ≈ 20 GB in ~/chefbot_collect/data/stream.
- USB: kernel logs "usb 2-3-port3: Cannot enable. Maybe the USB cable is bad?" every ~4 s since 12:40 on the 10G hub the pole
  D455 sits behind (port 2); likely the wrist D405's cable failing its USB 3 link on hub port 3 (inference). Recorder now restarts
  a stalled stream and logs it (4736aab).
- D455 IMU: kernel IIO devices present (accel/gyro via usbhid on the D455), but raw reads are 0 and no library gets samples.
  On Nov 3 the viewer streamed it with the D455 directly on bus 004 (no hub). Untested hypothesis: the hub.

## 2026-11-05 13:00–14:30 — kitchen map and work-section driving (branch `lateral-speed`, local, 4cef95b)
- Robot runs `~/chefbot_speedexp` base.sh (cap 0.5 m/s, accel/brake 0.25). Mocap descriptions captured by pausing the stack
  (~30 s each, no motion): ~/chefbot_collect/motive_descriptions_20261105*.log.
- maps/src_kitchen from Motive bodies (all turned −2.96° from Motive x: the room vs Motive's ground-plane axes): island
  (26), L leg 1 + stove (27), L leg 2 (28, depth assumed), fridge (28's 1.79 m markers), far counter (29), test counter (25).
  No-go: floor west of the island, edge 16 in off its end, floor level (arm may reach over). Fridge aisle: heading
  zone (heading_zones.yaml): base faces along it ±20° only — robot/nav/footprint.py heading zones (planning + guard).
- Jack's gamepad drive (mocap log ~/chefbot_collect/mocap_markers_push_20261105.log): plate 8–10 mm from island and stove
  counter without contact (edges good to ~1 cm); gripper ~5 cm from the fridge where the arm model says 0 (≥5 cm cautious).
- Parks (recorded, ~/chefbot_collect/data/stream/k*): k1_island 8/8, k1_stove 8/8 (0.38 m, teleop); k2_alt_fast 2 parks
  then guard stop (arm outline into the no-go mid-turn); k3_alt_fast 10/10 (0.42 m, exp50); k4_alt_fast 10/10 (0.46 m,
  exp50, all rules). Two guard stops were correct (locked arm 0.68 m sweeping into no-go / fridge aisle during 180° turns):
  turning targets kept in x 1.0–1.5. Island sink 35 in from its fridge end: no targets there.
- park: --sides, --x-range, --alternate COUNTER SIDE. Map page: https://claude.ai/artifact/WeFCEfWwwk7ZnsPgKA8wbx
- 14:40 rocking at arrival fixed: park --hold (RobotConfig.arrive_hold) freezes the goal in odometry once inside 3 cm / 3°.
  k5_alt_hold (same targets as k4): 74 s vs 99 s, settle 9 vs 34 s, reversals 11 vs 165, final ≤ 5.8 mm / 1.3°. Commit on lateral-speed.
  Sink drawn as the island's east 35 in, full depth.
- 14:45–15:20 around-the-island: k6–k8 partial (turn needed inside the heading-locked gap), k9 8/8 with back-off-and-turn
  (228 s), k11 8/8 via routes.yaml waypoints, stop-and-go (~26 s per trip). Pole camera restarted twice in k11 (USB).
  Smooth go_through clips the island in sim. Jack (15:20): stop hand-tuning routes; optimize random A to B, fast.
- 15:30–16:10 random A to B (the mapping cycle), branch lateral-speed (local): path style (cbe3b7c + speedups): clearance-aware
  A* (--clearance 0.7: measured from the base centre, whose floor is ~0.42), turns where there is room (--turn-in-room),
  ~180 deg turns tried both ways, search on turn ends when the sweep closes the aisles; --mix COUNTER:SIDE targets.
  Sim: around-island path gap-middle (x 2.98 vs 2.77), mid-path centre clearance 0.52 vs 0.35 m.
  m1_random: 5 reached, then a server hold (client silent 0.3 s; Wi-Fi or laptop load, not established).
  m2_random: 10/10 recorded, no holds; round-the-island trips 13.4–15.8 s, short ones 1.4–5.2 s, ≤ 8.4 mm / 1.2°.
  Planning a 180° cross-island trip takes ~6 s on the Mac (pure-Python A* repairs): worth speeding up.
- 15:20–15:50 data: drive loop moved ONTO the robot (~/chefbot_nav = 4e3fd15 + later real_nav.py; env chefbot-movement
  + mujoco/mink/coal/ompl pip-installed) after Mac↔robot Tailscale RTT spikes (to 474 ms) held the robot repeatedly.
  Recorder: motion-only, 1 frame/s, color+depth+IR, all mapped bodies. Old sessions (77 GB) deleted on Jack's word.
  Passes (copied to Claude_Code/robot_data/): p1_layout2 20/20, p2_layout3 20/20, p3_moved_unseen 30/30 (pole USB
  dropped 14x). No-go edge 20 in off the island. park: --mix, --end-margin, retries after link/mocap dropouts.
- 16:00 PR #2 https://github.com/otaylor2023/ChefBot/pull/2 (kitchen-nav → movement, stacked on PR #1): today's kitchen
  work without the 0.5 m/s cap (that stays on speed-exp). Runbook: docs/kitchen-runbook.md (in the repo). Local branch
  lateral-speed = kitchen-nav + the cap commit (what the robot ran). Pole camera calibration STALE since the 11-05
  re-angle: needs the teammate's calibration/pole.py (moves the arm); recorded data reprojects offline with the new T.
- 16:40 end of day: robot stack shut down (16:05), tunnel and SSH control connection closed. Pages updated. CLAUDE.md
  now opens with a "START HERE" pointer block (backup: CLAUDE.md.bak-2026-11-05).

## 2026-11-06 (offline: kitchen memory map with YOLOE, branch memory-map off kitchen-nav ca6f5ae, uncommitted)
No robot. Code in chefbot/perception/ (worktree mac_chefbot-movement, branch memory-map); batcomputer copy
.../Engineering_AI_Agents/ChefBot_memmap (rsync, no .git). Tests: 16 new pass (perception); full suite on batcomputer
465 passed, 4 failed = the same 4 as kitchen-nav there (autoresearch generators x3, test_real_host table).
- Pole camera -> world chain (pole_geometry.py) checked against the map, no arm (chain_check.py): with the 11-05 pole T,
  the ~18 cm arm-base vs turning-centre offset is REAL (points at counter-top height inside their boxes: 29% with no
  offset, 73% with the assumed 0.1825 m, 91% fitted). Fit (p1+p2, 32 frames): dx 0.198 dy 0.014 dz 0.088 m, roll
  0.78 deg -> island top 0.908 (0.91), stove counter 0.843 (0.83). Held out p3: inside 0.89, mean |dz| 0.034 m.
  Stored in chefbot/perception/pole_mount_fit.yaml (only for that T). Marker-PnP check (IR) not done. Probe images:
  projected entries land on the objects (tomato at 1.09 m, tennis ball at 0.70 m).
- Detector: YOLOE-26 (ultralytics 8.4.204). batcomputer vision_venv (/mnt/d/.../vision_venv): torch 2.9.1+rocm6.3 on the
  RX 6800 (ROCm 6.3 was already installed). Mac: ~/.venvs/chefbot_vision (torch 2.15 MPS). Per 1280x800 frame:
  RX 6800 s@1280 13.6 ms (fp16 11.5); M1 MPS s@1280 136 ms, n@1280 57 ms, s@640 37 ms; CPU s@1280 ~320 ms (both).
  1280 finds ~40% more than 640. Prompt-free (-pf) labels are junk for this (verbs, "water tank"); text prompts from
  vocab_kitchen.txt (DRAFT, Jack to prune) give tomato, tongs, spoon, lego brick, controller, sponge ... whisk weak.
- Memory map (memory_map.py): object = 3D cluster on a counter top (stickers, wall things rejected), label by vote; gone =
  depth ray reaches the bare counter top at its spot twice -> position unknown; through-the-top depth (stove's black
  glass) is no evidence. p2 -> p3 replay: tomato lost from the stove counter (n=28) and found on the island (n=66,
  "seen as bowl at first"); tongs moved along the stove counter. Not scored: needs Jack's object list + what moved.
  Maps: batcomputer chefbot_data/detections/txt_s_1280/map_*.json (copies in Claude_Code/robot_data/detections/).
- goto (real_nav goto NAME --map maps/src_kitchen [--objects F]): looks the object up, parks squared up to the nearest
  side; deep objects move to the nearest legal spot (reported as untrained reach). Resolved offline for tomato, tongs,
  sponge, bowl ...; not driven.
- Training data: export_yolo.py -> chefbot_data/datasets/pseudo_txt_s_1280/<pass>/ (pseudo-labels, uncorrected) +
  data.yaml split (train p1 p2, val p3).
- 2026-11-06 late: NAMING moved off the detector. name_objects.py names each map object once from its 4 best views (score x
  sharpness / distance), with a VLM: default gemini-3.5-flash-lite (API, GEMINI_API_KEY in Claude_Code/.env, 4 parallel), or
  ollama qwen3-vl:8b-instruct (local, free, batcomputer). Prompt asks for the whole object (not a logo/part/sticker).
  Eval (eval_naming.py, 75 distinct objects over 3 layouts; answer keys = Claude's reading of the strips, robot_data/naming/ref_*.json):
  gemini-4-argon 69/75 $0.0036/obj; gemini-3.1-pro 67 $0.005; gemini-3.8-flash 67 $0.003; gemini-3.5-flash-lite 67 ~$0.0005;
  robotics-er-2 66; qwen3-vl local 66 (free, ~2 s/obj); YOLOE labels alone ~50%. Nearly all misses: the black two-armed tool
  (models: spring clamp / hand gripper; key says tongs - ASK JACK), a small dark object behind the controller (key: phone?,
  unclear), the moved tomato (one far view: "apple"). Whole map named in 12 s on the Mac. Paid spend for all tests ~$1.7.
  Also: parking only from floor-reachable sides (goto.Floor; 4,393-spot sweep clean), duplicate entries merged by overlap.
