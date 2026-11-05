# ChefBot hardware tests

One entry per test: date, commit, command, settings, expected, observed, limitations.
Tests ran from branch `movement` before it was rebased onto main 8bdd32d. The commit hashes below are the pre-rebase
ones, kept reachable by the tag `archive/movement-2026-11-03`; the robot ran them from `~/chefbot_movement`.
Paths: `runs/` is the laptop checkout's (gitignored, not in the repo); `docs/evidence/` is in the repo.
`chefbot` is an SSH alias for tidybot01.

---

## HT-0a: live mocap pose on the laptop, no motion (stage 0)

**Status:** PASSED 2026-11-03 13:06:52–13:07:52 PST (one run).
**Commit:** `85c2ef2` (movement), on top of the robot snapshot `54a0ca4`.
**Moves anything:** no. No base driver, no gripper driver, nothing commanded. The onboard server writes no
command keys for a client that never commands (tested: `test_a_client_that_only_watched_leaves_the_drivers_alone`).

### Preconditions
- Teammate told; nobody else running mocap.sh or an onboard server (`ps`, port 8765 free).
- Windows / Motive: Streaming on, Transmission Type **Unicast**, rigid body `chefbot-tidy` enabled and tracked.
- Robot on SRC Wi-Fi (`wlp88s0` 172.24.x).
- Robot stationary. Arm: whatever state the teammate left it in; this test does not touch it.

### Deploy (robot), separate from the teammate's `~/ChefBot`
```
# Mac
git archive 85c2ef2 | ssh chefbot 'mkdir -p ~/chefbot_movement && tar x -C ~/chefbot_movement'
# robot
~/miniforge3/bin/conda create -y -n chefbot-movement python=3.11
~/miniforge3/envs/chefbot-movement/bin/pip install numpy msgpack msgpack-numpy "websockets>=13" pyyaml redis
```

### Run
| Where | Command |
|---|---|
| robot, terminal 1 | `cd ~/chefbot_mocap && bash ~/chefbot_movement/chefbot/onboard/bringup/mocap.sh --quiet` |
| robot, terminal 2 | `cd ~/chefbot_movement && ~/miniforge3/envs/chefbot-movement/bin/python -m chefbot.onboard.server` |
| Mac | `ssh -o ControlPath=~/.ssh/cm-chefbot -O forward -L 8765:localhost:8765 chefbot` |
| Mac | `python -m chefbot.scripts.real_pose --seconds 60` |
| robot, at ~40 s | stop terminal 1 (Ctrl-C on mocap.sh) |

### Pass criteria
1. Lines read `LIVE` with a plausible pose (the Nov 2 stale value was x 0.395, y 0.403; it should differ or at least jitter).
2. At rest: live ≥ 99% before mocap is stopped; spread ≲ 1 mm in x, y and ≲ 0.1° in yaw.
3. After stopping mocap.sh: `stale` within ~0.3 s (`mocap_stale_s` 0.25 + one state period), and stays stale.
4. Server log shows no `halted` line; Redis command keys (`hb1::desired_pose`, `hb1::stop`, OpenSai goals) unchanged.

### What it would not show
- Whether yaw matches the base's heading (`marker_yaw_offset_deg` is still 0.0: stage 1).
- Freshness while the robot moves, Motive's true rate (the server samples at 50 Hz), or Wi-Fi dropout behaviour.

### Observed (2026-11-03, run by Claude Code from Jack's Mac)
- Deployed: `git archive 85c2ef2` → `tidybot01:~/chefbot_movement` (`DEPLOYED_COMMIT` = 85c2ef2); conda env
  `chefbot-movement` (Python 3.11.18, numpy 2.4.6, websockets 17.4, redis 8.1.0). Teammate's `~/ChefBot` untouched.
- Context: teammate's arm stack running (Franka driver + OpenSai `chefbot_fr3.xml`); no base driver, no gripper driver.
- Probe (5 s of `mocap.sh`, no --quiet): Motive 3.3.0.1, NatNet 4.2.0.0, "Mocap Framerate 120.00", one rigid body
  `chefbot-tidy` ID 23, 595 frames, `Tracked=1`, mean marker error ≈ 1.04 mm.
- Run: `real_pose --seconds 60`; mocap client (PID 10599) stopped with SIGINT at robot 13:07:32.65.

| Criterion | Result |
|---|---|
| 1. LIVE with a plausible pose | yes: x −0.6426, y 0.5837 (z 0.447 raw), yaw −0.54° (Nov 2 stale value was x 0.395, y 0.403) |
| 2. live ≥ 99%, spread ≲ 1 mm / 0.1° | 1675/1675 samples live while publisher up (100%); std x 0.02 mm, y 0.01 mm, yaw 0.003°; peak-to-peak 0.11 mm, 0.08 mm, 0.016° |
| 3. stale within ~0.3 s, stays stale | pose last changed t = 39.795 s; first stale sample t = 40.060 s → **0.265 s**; stale for all remaining 20 s |
| 4. no halt, command keys unchanged | server log has no `halted` line (also after a probe connect/disconnect and after Ctrl-C); 9 command keys identical before/after |

Other numbers: mocap age while up median 20 ms, max 40 ms (server samples at 50 Hz; client got 42 samples/s);
state message age max 45 ms over the tunnel; first state was `stale` (the Redis value predating the server) and
turned LIVE at t = 0.05 s.

Evidence (Mac): `runs/ht0a/` — `real_pose.jsonl` (sha256 6dacadf1…bc1f66d),
`real_pose_console.txt`, `server.log`, `mocap.log`, `ht0a_mocap_probe.log`, `ht0a_keys_before.txt`, `ht0a_after.txt`.
Robot copies in `~/chefbot_movement/runs/ht0a/`.

### Findings from this run (not failures of HT-0a, but they block later stages)
1. **Stale odometry looks live too.** `hb1::current_pose` exists in Redis with no base driver running (left from an
   earlier run, persisted by Redis), so `read_state` "localized" mocap against a frozen odometry pose
   (`odom->world` column). Needs the same freshness treatment as mocap before stage 2/3.
2. ~~Motive's up axis is probably Y, not Z.~~ **Retracted 2026-11-03 ~13:40: the stream is Z-up.** The inference came from the
   rigid body's *description* (marker offsets with local y ≈ 0), which is apparently in Motive's internal Y-up convention, not the
   streamed frame. Frame data decides it: between HT-0a and the HT-1a probe the base was driven (G1) and turned 9.2° about axis
   (0.019, 0.033, −0.999), i.e. z; x and y moved 0.72 m and 0.20 m while z stayed 0.447 → 0.443 m. `frames.yaw_of_quat_xyzw`
   (z up) is the right formula. The teammate's "0 or 180" is then about the fixed marker offset, as the handoff said.

---

## HT-1a: mocap frame from driving by hand (stage 1)

**Status:** PASSED 2026-11-03 13:35–13:38 PST (one session). Result committed as `ee7cb88` (onboard/config.yaml).
**Commit:** `97f8044` (movement), deployed to `tidybot01:~/chefbot_movement`. `--echo` smoke-tested on the robot 2026-11-03: F710 found (6 axes), no motion.
**Moves:** yes, the base only, by Jack with the gamepad at ≤ 0.2 m/s and ≤ 0.5 rad/s. No autonomous commands; the recorder only reads Redis.

### Questions it answers
1. Motive's up axis (expected: y), from the spin's rotation axis and from the plane of all positions.
2. `marker_yaw_offset_deg`, estimated separately from forward, back, left and right; they should agree within a few degrees.
3. Whether the floor frame is mirrored (left/right 180° off forward/back).
4. `marker_xy`, from the circle traced in a left spin; and that a left spin reads as counter-clockwise.

### Preconditions
- Teammate agrees. **Their arm stack is running (arm driver + OpenSai holding the arm); decide with them whether to stop it and lock the joints in Desk before the base moves.**
- Clear floor ≥ 1 m around the base; one person on the motor e-stop remote, per the lab instructions.
- Base motor power on. Nothing else holding the CAN bus (no `redis_driver.py`).
- Motive streaming (Unicast, `chefbot-tidy` tracked).

### Before recording
- G0 (no motion): `cd ~/chefbot_movement && ~/miniforge3/envs/tidybot2/bin/python -m chefbot.onboard.bringup.gamepad --echo`.
  Push the stick diagonally, up+right, up+turn: the arrow must show one motion only. Ctrl-C.
- G1 (motion, powered, 0.2 m/s): run the gamepad, Start, hold left bumper, a short tap each way; check that "forward" goes the way
  the arm faces, "left" goes to the arm's left, "turn left" is counter-clockwise from above. Back to stop.

### Run (robot terminals; "forward" = the way the arm faces)
| Terminal | Command |
|---|---|
| A (Claude, background) | `cd ~/chefbot_mocap && bash ~/chefbot_movement/chefbot/onboard/bringup/mocap.sh --quiet` |
| B (Jack: gamepad) | `cd ~/chefbot_movement && ~/miniforge3/envs/tidybot2/bin/python -m chefbot.onboard.bringup.gamepad` |
| C (recorder, one per move) | `cd ~/chefbot_movement && ~/miniforge3/envs/chefbot-movement/bin/python -m chefbot.onboard.calibration.base record runs/ht1a/<move>.jsonl` |

The gamepad runs in one-axis mode by default: the left stick snaps to its larger axis and turning is ignored while
translating, so an imperfect stick still commands a pure move. For each of `forward`, `back`, `left`, `right`: start the
recorder with the robot still, hold the **left bumper** (base frame), push the stick in that direction 0.4–1 m, stop, then
Ctrl-C the recorder. Moves that still turned > 5°, curved, or went < 0.2 m come out `ok: false` and are not used. Then `spin`: rotate in place to
the left (counter-clockwise from above), slowly, as far round as is comfortable (a full turn is best).

### Solve
```
~/miniforge3/envs/chefbot-movement/bin/python -m chefbot.onboard.calibration.base solve \
  --forward runs/ht1a/forward.jsonl --back runs/ht1a/back.jsonl --left runs/ht1a/left.jsonl \
  --right runs/ht1a/right.jsonl --spin runs/ht1a/spin.jsonl
```
Pass: `up_axis_agree` true; per-move offset disagreement ≤ ~5°; `mirrored` false; `spin.turned_left` true;
`marker_xy_std_m` ≲ 1 cm.

### Observed (2026-11-03)
- G0 (stick echo) and G1 (powered direction check): passed, per Jack. Jack saw a brief lurch as the swerve modules
  re-steer at each change of direction; the solver fits only the steady part (skips the first 0.15 m and last 0.05 m, `d1dc224`).
- Recordings (robot `~/chefbot_movement/runs/ht1a/`, copied to Mac `runs/ht1a/`; sha256 in
  `docs/evidence/ht1a/sha256.txt`): `forward` 868 live samples, `backward` 751, `left` 442, `right` 1029, `clock-wise` 1866.
  The spin was clockwise (to the right), not left as the procedure asked; the solver handles either.
- Solve (`docs/evidence/ht1a/solve.json`):

| Question | Result |
|---|---|
| Up axis | **z**, both ways: spin rotation axis 0.8° off z (1548 samples), position-plane normal 0.4° off z (height spread 0.5 mm) |
| Heading offset per move | forward 2.78°, left 3.70°, back 3.01°, right 3.06° → **3.14°**, each within 0.6° of the mean |
| Mirrored | no |
| Spin | 379° clockwise; `turned_left` false, consistent with clockwise (yaw sign right) |
| marker_xy | **(0.0719, −0.0018) m**, std 1.0 mm; circle radius 71.9 mm, rms 1.0 mm |
| Move quality | all four `ok`; heading drift during moves 0.4–3.1°; straightness ≤ 0.009 |

- Check (`docs/evidence/ht1a/validate.txt`), `frames.base_from_mocap` with the solved values vs. the old config (0°, [0, 0]):
  base centre through the spin 1.4 mm rms / 3.6 mm max (old: 69 mm rms); each move's travel vs. base heading + move
  direction within 0.55° (old: 2.8–3.7° off).
- Mocap stopped after the session.

### Limitations
- One session, one floor area, ≤ 0.87 m per move; the offsets are validated on the same data they were fitted to
  (the spin check is independent of the moves, and vice versa, but not of the session).
- `marker_xy` is relative to the base's rotation centre in a spin, assumed to be the base frame origin the driver uses.
- Wheel slip is in the data; at 0.2 m/s it looks small (straightness ≤ 0.009).

---

## HT-2a + HT-3a: first autonomous moves, A → B by mocap (stages 2–3)

**Status:** HT-3a first moves PASSED; HT-2a timeout fired but stop not measured (re-run). 2026-11-03 13:58–14:00 PST.
**Commit:** `ad47a92` (movement), deployed to `tidybot01:~/chefbot_movement`. Non-moving checks on the robot passed
2026-11-03: `base.sh` syntax, driver refuses out-of-range limits, `base_controller` imports in `zitibot`, server imports.
**Moves:** yes, the base, autonomously, ≤ 0.1 m/s and 0.3 rad/s (driver cap 0.125 m/s, 0.4 rad/s); straight line, no obstacle avoidance.

### Safety layers in play (each stops the base on its own)
| Layer | Stops when | Verified |
|---|---|---|
| Person on the motor e-stop remote | anything looks wrong | — |
| `drivers/base/driver.py` | no new `hb1::cmd_seq` for 0.5 s; `hb1::stop`; kill | unit tests; **HT-2a** on hardware |
| `onboard/server.py` watchdog | client silent 0.3 s / disconnects after commanding | unit tests |
| `robot_io.base_target` | refuses without live mocap + live driver | unit tests |
| `scripts/real_nav.py go` | lost mocap/driver, halt, error, > 0.15 m off line, moving away, timeout, Ctrl-C | sim tests; **HT-3a** |

### Preconditions
- **Teammate's arm stack stopped and joints locked in Desk** (`base.sh` refuses while `sai_franka_robot_redis_driver` runs).
- Clear floor ≥ 1 m around A, B and the line between them. One person on the motor e-stop remote. Base motor power on.
- Motive streaming (Unicast, `chefbot-tidy` tracked). Gamepad script not running (it holds the CAN bus).

### Steps
| # | Where | Do | Expect |
|---|---|---|---|
| 1 | robot T1 | `cd ~/chefbot_movement && chefbot/onboard/bringup/base.sh` | "1) base driver… 2) mocap… 3) onboard server… serving" |
| 2 | Mac | `ssh -o ControlPath=~/.ssh/cm-chefbot -O forward -L 8765:localhost:8765 chefbot` | — |
| 3 | Mac | `python -m chefbot.scripts.real_pose --seconds 5` | `LIVE`, odom->world localized |
| 4 | Mac | `real_nav mark A`, then `real_nav point S1 --from A --forward 0.3` | A saved; S1 0.3 m ahead |
| 5 | Mac | `real_nav go S1` (Enter) | ~0.3 m forward, `reached`, error < 3 cm |
| 6 | Mac | `real_nav go A` | back to A |
| 7 (HT-2a) | Mac + robot | `real_nav point S2 --from A --forward 0.5`; `go S2`; ~2 s in, Claude runs `kill -9` on the server process | base brakes and holds within ~0.5 s (driver timeout); `go` reports lost driver/server |
| 8 | robot T1 | Ctrl-C, then restart `base.sh`; Mac: `go A` | back to A |
| 9 (Jack's A→B) | — | Ctrl-C `base.sh`; gamepad the base to B; stop the gamepad; restart `base.sh`; `real_nav mark B`; `go A`; `go B` | both reached |

Pass: every `go` ends `reached` with final error < 3 cm / 3°, no stop besides HT-2a's, and in HT-2a the base stops
within ~0.5 s of the server's death (from `runs/real_nav/*.jsonl` and the driver log).

### Observed (2026-11-03, commit ad47a92 on the robot; Claude ran the laptop side, Jack on the e-stop)
- Pre-checks: arm driver stopped (PID 6655 gone), `base.sh` up (driver heartbeat ~97 Hz, "ready: idle until a fresh
  command"), pose live and localized (A ≈ x 0.140, y 0.394, yaw −46.8°; jitter 0.04 mm). Dry run of `go S1` cancelled
  without moving; `hb1::cmd_seq` empty beforehand (nothing commanded).

| Move | Distance | Time | Final error | Max off line | Result |
|---|---|---|---|---|---|
| A → S1 (0.3 m forward), 13:59:11 | 0.300 m | 4.6 s | **4.2 mm, 0.47°** | 3 mm | reached |
| S1 → A (0.3 m back), 13:59:28 | 0.298 m | 4.5 s | **6.7 mm, 0.06°** | 3 mm | reached |
| A → S2 (0.5 m), server frozen (SIGSTOP) 2.5 s in, killed 3 s later | 0.136 m done | 2.8 s | — | 5 mm | stopped: state stale (reported as "lost mocap", fixed in 63dab6f) |

- HT-2a: the driver log shows `[base] no fresh command for 0.50 s: braking` before `killed`, so the timeout fired with
  the server frozen. **Not measured:** stop latency and distance; the robot-side recorder's 12 s window ended before the
  freeze (my timing error). Re-run with a longer recording.
- Logs: Mac `runs/real_nav/` (A, S1, points.json, per-move JSONL), `runs/ht2a/` (driver/server/mocap
  logs from base.sh, freeze timestamps, the recorder file).

---

## HT-3b: characterization session in a fenced square (stage 3)

**Status:** PASSED 2026-11-03 14:24–14:29 PST: 52/52 legs reached. Robot released to the teammate afterwards. **Commit:** `a97e144`.
**Moves:** yes, autonomously, ≤ 0.12 m/s and 0.3 rad/s, base centre within ±`--half` of the start pose C.

- Plan: `python -m chefbot.scripts.real_characterize plan --half 0.2` → 26 moves (repeat ×5 at 0.16 m, 8 directions,
  turns ±15/45/90/180°, 3 step+turn combos, speeds 0.05/0.08/0.12 m/s), each out from C and back (52 legs, ~15 min).
- Run (Mac): `real_characterize run --half <h>`; Claude also runs `calibration/base.py record` on the robot for the
  whole session (raw mocap and odometry at 100 Hz, independent of the laptop).
- Report: `real_characterize report runs/characterize/<session>` → `report.txt` in that folder.
- Open question before running: does the 3 ft square bound the whole base (then `--half` ≈ 0.91/2 − base half-width,
  turns sweep the base's corners further) or only the base centre?

### Observed (2026-11-03, commit a97e144 on the robot; `--half 0.2`, the whole base inside the 3 ft square)
- C: x 0.285, y 0.276, yaw 179.6°. Arm stack off. `base.sh` started by Claude; robot-side recorder at 98 Hz for 317 s.
- **52 of 52 legs `reached`**; no fence, corridor or timeout stops.

| Block | Final error mean / max | Notes |
|---|---|---|
| repeat (5× 0.16 m fwd, out+back) | 1.0 / 1.9 mm, yaw ≤ 0.17° | endpoints spread (std) **0.2 mm** each axis; mean offset (−1.8, +0.4) mm |
| directions (8) | 3.8 / 8.3 mm, yaw ≤ 0.67° | worst forward-right (+6.0, −4.0) mm; no common offset across directions |
| turns (±15/45/90, 180) | 0.9 / 1.7 mm, yaw ≤ 0.59° | yaw overshoot ≤ 0.9°; centre drift during turns mean 5.5, max 14.9 mm (robot-side) |
| combo (step + turn) | 1.8 / 3.4 mm, yaw ≤ 0.82° | max 11 mm off the line while turning and moving |
| speed (0.05/0.08/0.12) | 2.6 / 3.6 mm | no overshoot beyond 1.9 mm; time to 5 cm 3.4 / 2.3 / 2.1 s |

Robot-side (100 Hz, robot clock; `docs/evidence/ht3b/robot_side_analysis.txt`):
- True peak base speed **0.167 m/s** (99th pct 0.110) > the 0.125 m/s nominal: the driver's limit is per axis (x, y), so a
  diagonal can reach ~0.177. The laptop report's 0.207 m/s was an artifact of differentiating at arrival times.
- Wheel odometry / true distance on straight legs: **0.998 ± 0.006** (n 32). Rotation ratio 0.88 ± 0.49 (n 16): too noisy
  to use (segmentation), not a finding.
- Odometry alone, untied from mocap, drifted **42 mm and 2.7° in 5 min**: why mocap closes the loop.
- The laptop report's "odometry/mocap = 1.000" is meaningless (that odometry is re-tied to mocap every read); fix in the report.

Problems found:
1. **`base.sh` did not stop** on SIGINT/SIGTERM: cleanup SIGINTed a background job (ignored) and waited forever; driver
   stayed up (idle, braking). Stopped by hand (hb1::kill, then TERM). Fixed and tested on the robot with stand-ins in `dad29a2`.
2. The robot-side recorder also ignored SIGINT (background job); stopped with TERM.
3. Per-axis speed limit lets diagonals go √2 faster than intended.

Evidence: `runs/characterize/20261103_142432/` (plan, results, 52 per-leg logs, report),
`runs/ht3b/` (robot recorder, base.sh logs); sha256 in `docs/evidence/ht3b/sha256.txt`.

---

## HT-4a: around the counter top (stage 4)

**Status:** PASSED 2026-11-03 15:51–15:53 PST: hugging the counter's near side, 3/3 legs. **Commit:** `f5ca190` on the robot.
- Map `maps/src_counter` (counter from rigid body `Chef-botcounter-top`; bounds PROVISIONAL → replace first).
- Dry run against `onboard/sim.py` (`docs/evidence/counter/sim_lap.txt`): far_side → east_end → near_side → west_end →
  far_side, all reached; 1394 poses, base centre ≥ 0.458 m from the counter box.
- Plan on the robot: `base.sh`; real_pose; `real_nav go near_side --map maps/src_counter` (no detour, short), then
  `west_end`, then `far_side` (around the end), at 0.1 m/s, Jack on the e-stop, robot-side recorder running.
- Pass: every leg reached, no `footprint reached an obstacle`, base centre ≥ 0.43 m from the counter in the robot-side log.

### Observed (2026-11-03, Jack on the e-stop, Claude running the laptop side)
- Teammate's stack stopped first (it had started at 15:24 and base.sh correctly refused to run beside it); arm locked,
  joints static: q = [-0.033, -0.829, -0.003, -2.320, -0.090, 2.334, 0.859] → arm reaches 0.54 m ahead, tip at z 0.94 m
  (counter top 0.93 m) → footprint 0.550 m (from the last joint readings; arm driver off). Margin 0.10.
- Counter live from its rigid body at (1.4909, 2.0600), matching the map. Map bounds still provisional (ymin −0.05 from HT-3b).

| Leg | Path | Final error | Notes |
|---|---|---|---|
| start (−0.55, 0.64, 154°) → hug_a (0.50, 0.86, 0°) | 1.07 m, 2 segments | 0.3 mm, 0.07° | turned 154° while moving |
| hug_a → hug_c (2.40, 0.86, 0°) | 1.90 m, 2 segments | 6.7 mm, 0.01° | along the counter, facing along it |
| hug_c → hug_a | 1.90 m, 2 segments | 6.1 mm, 0.05° | driving backwards |

Robot-side 100 Hz (`docs/evidence/ht4a/clearance.txt`): 12,580 live samples over 128 s, 4.93 m travelled.
Base centre ≥ **0.713 m** from the counter box (plan 0.72; keep-out 0.65). Base circle gap ≥ 0.38 m. **Arm box (locked
pose) gap ≥ 0.31 m, 0 overlaps.** Along the counter y 0.846–0.867 (target 0.86). base.sh stopped cleanly on SIGTERM
(dad29a2 fix confirmed on hardware). Evidence: `runs/ht4a/`, `runs/real_nav/20261103_1551*`,
`…_1552*`; sha256 in `docs/evidence/ht4a/sha256.txt`.

Limits: one run; near side only (ends and far side need the real room bounds); a counter that is a box, measured from 4
corner markers; the arm's height not used (conservative).

---

## HT-4b: a full lap around the counter, faster, rotating to face each side (stage 4)

**Status:** PASSED 2026-11-03 15:58–16:01 PST, 5/5 legs. **Commit:** map `33deb88` (laptop), robot `f5ca190`.
**Settings:** driver 0.15 m/s, 0.5 rad/s (`MAX_VEL_XY=0.15 MAX_VEL_YAW=0.5 base.sh`); `go --speed 0.15 --yaw-speed 0.5
--margin 0.05`; footprint 0.550 m from the locked arm (tip at z 0.94 m, ~1 cm above the 0.93 m counter top: too
little, with an unverified mount height, to let the arm pass over the counter, so the arm stayed in the footprint).
Bounds xmax/ymax extended on Jack's go-ahead (not measured); Jack on the e-stop.

| Leg (counter-clockwise) | Path | Final error |
|---|---|---|
| hug_a → lap_near (face +y) | 0.99 m | 0.7 mm, 0.25° |
| → lap_east (face −x) | 2.67 m, 5 segments | 1.4 mm, 1.07° |
| → lap_far (face −y) | 2.65 m, 5 segments | 8.7 mm, 1.88° |
| → lap_west (face +x) | 2.62 m, 5 segments | 0.5 mm, 0.13° |
| → hug_a | 1.67 m, 3 segments | 5.5 mm, 0.17° |

Robot-side 100 Hz (`docs/evidence/ht4b/clearance.txt`): 20,519 live samples, 209 s, 10.79 m, 696° turned in total.
Base centre ≥ **0.625 m** from the counter (plan 0.65; keep-out 0.60). **Arm box ≥ 0.081 m** from it, 0 overlaps.
Base centre ranged x −0.29..3.29, y 0.86..3.19.
Speed: the driver's odometry peaked at 0.157 m/s and 27.7°/s (limits 0.15, 28.6); mocap saw short bursts up to
0.229 m/s and 38.4°/s (0.42% of the time > 0.16 m/s; at the peak the wheels read 0.113 m/s): the base itself lurching,
most likely as the swerve modules re-steer at segment changes, not a limit being exceeded by the driver.
Note: the first leg's result was read from its log because my shell loop's variable `path` clobbered zsh's PATH.

Limits: one lap; the far side and east end were driven into floor whose walls are not measured; final yaw errors up to
1.9° (the 3° tolerance) are larger than in slower runs; corners cut the plan's 0.65 m by 2.5 cm.

### HT-4b, observed from outside (Jack, on the e-stop, 2026-11-03)
- The arm was in a safe position throughout, not ~1 cm from the counter (the 8 cm figure is the arm's box
  seen from above, conservative by design).
- The base could have come closer to the counter, and could move a bit faster; the test was fine.
- In turns it occasionally came close to the counter's corners, but nothing would have hit.
- **There is a wall at the far-left corner** (back, behind the counter on the left): the 360° turn there was
  close. Stay closer to the counter on that side. (Wall position not yet in the map.)

---

## HT-4c: the stop-distance re-run, then the closer lap with the arm over the counter

**Commit:** `c30cf93` (deployed). Arm locked as in HT-4b (tip 1.047 m with the 0.44 m mount). Walls in the map.
- First: **HT-2a re-run** (driver-timeout stop distance) at 0.15 m/s, since the counter gap drops to ~0.12 m.
- Then `close_near → close_east → close_far → close_west → hug_a` (9.39 m), 0.15 m/s, margin 0.05: base centre 0.45 m
  from the counter's edges (base body ~0.12 m from it), arm over the counter, ≥ 0.85 m from the east wall.
- Pass: all legs reached; robot-side log: base circle never inside the counter box; arm-moved stop never fired.

### Observed (2026-11-03 16:56–17:00, after the robot's 16:40 reboot; Jack on the e-stop)
- Robot back on SRC, idle; arm static in a new pose q = [0.048, −1.082, −0.001, −2.227, 0.030, 2.206, 0.737] (last values in
  Redis, arm driver off): tip at z 1.173 m, 24 cm over the counter top; footprint 0.445 m vs walls, 0.359 m vs the counter.

**HT-2a re-run (stop distance):** driving toward hug_a at 0.15 m/s, the server frozen (SIGSTOP) at robot t 1793753795.667.
Driver log: "no fresh command for 0.50 s: braking". Speed at the freeze 0.119 m/s; at rest (< 5 mm/s) 1.11 s later, after
**66 mm** and 11.5°. The laptop saw the silent server in 0.54 s. Worst case at 0.15 m/s with the driver's 0.1 m/s²
braking: ≈ 0.5 s × 0.15 + 0.15² / 0.2 ≈ 0.19 m → stronger braking on timeout before going faster or closer.

**Close lap:** hug_a → close_near → close_east → close_far → close_west → hug_a, 6/6 reached, final errors 0.1–3.5 mm, ≤ 0.58°.
Robot-side: 214 s, 10.33 m, 755° turned. Base centre ≥ **0.426 m** from the counter (body gap ≥ **9.6 cm**). Arm parts
lower than the counter top + 5 cm: gap ≥ 9.9 cm. Higher arm parts (lowest 21 cm above the top): ≥ 5.4 cm horizontally
(did not even pass over). East wall ≥ 0.83 m, far wall ≥ 1.31 m. Wheels' odometry peaked 0.155 m/s; mocap saw a lurch
to 0.298 m/s (marker 0.291 while wheels read 0.144; 0.26% of the time > 0.17 m/s).
Evidence: `docs/evidence/ht4c/` (stop_analysis.txt, lap_clearance.txt, sha256.txt), `runs/ht4c/`.

---

## HT-4d: harder braking, smooth following, square stations

**Status:** PASSED 2026-11-03 17:13–17:20 PST (one network stall held the robot once; leg retried). Robot at `a9919f5` + map `6d1cc18`.
1. Stop test again (freeze the server at 0.15 m/s): expect at rest in ≲ 0.6 s, ≲ 9 cm.
2. Station lap: hug_a → counter_south → counter_east → counter_north → counter_west → hug_a, smooth, 0.15 m/s, margin 0.05.
Pass: braking within the estimate; every station reached square (yaw error ≤ 1°); no footprint / arm-moved stops; robot-side
clearances as HT-4c or better.
Sim dry run (docs/evidence/counter/sim_station_lap.txt): 5/5, ≤ 22 mm off the path.

### Observed (Jack on the e-stop)
- Arm static, new pose (tip 1.237 m, 31 cm over the counter): footprint 0.506 m vs walls, 0.330 m vs the counter.
- First start refused ("touching an obstacle where it stands"): the base stood at x −1.01, inside the provisional west edge
  (x −1.40) grown by 0.506 m. Nothing commanded; the pre-scheduled server freeze then shut base.sh down cleanly. West edge
  moved to x −2.0 (open, per Jack) in `6d1cc18`.

**Stop test:** server frozen at 0.131 m/s. Driver: "no fresh command for 0.30 s: braking". At rest (< 5 mm/s, < 1°/s) 0.90 s
after the freeze, **78 mm**, 11°. Predicted with the new settings 0.131 × 0.4 + 0.123² / 0.5 ≈ 8 cm; with HT-4c's settings
≈ 16 cm from this speed. Worst case at 0.15 m/s now ≈ 10.5 cm (was ≈ 19).

**Station lap** (smooth following, carrot 0.12 m; square stations from the counter box, 0.25 m straight approach):

| Leg | Path | Final error | Max off the path |
|---|---|---|---|
| → hug_a | 0.68 m | 1.6 mm, 0.26° | 10 mm |
| → counter_south | 1.24 m | 3.2 mm, 0.53° | 14 mm |
| → counter_east | 2.64 m | 1.5 mm, 0.20° | 20 mm |
| → counter_north | 2.59 m | 4.2 mm, 0.21° | 23 mm |
| → counter_west | 2.60 m | **held by the server's watchdog** ("no word from the client for 0.30 s"), 27 mm short of the approach point | 19 mm |
| → counter_west (retry) | 0.28 m | 1.5 mm, 0.04° | — |
| → hug_a | 1.44 m | 5.1 mm, 0.00° | 14 mm |

Square at the stations (robot-side, 1 s at rest): heading within **0.54°** of square, within **4 mm** of the station, at
most **2.9 mm** along the edge. Robot-side: 199 s, 11.30 m; base body ≥ **6.6 cm** from the counter (centre ≥ 0.396 m;
planned keep-out 0.38); peak 0.208 m/s (99th pct 0.154).
Watchdog halt: the laptop's loop never paused (≤ 63 ms between samples), but the state it received aged to 277 ms: a
~0.3 s **Wi-Fi stall** between the Mac and the robot (Tailscale direct over SRC at the time, 12 ms). The halt is the
designed response; for operation the motion loop should not depend on that link.
Evidence: `docs/evidence/ht4d/` (stop_analysis.txt, lap_analysis.txt, sha256.txt).

---

## HT-5a: counter station lap after the rebase onto main 0a3c7d7, with pole-camera images at each station

**Status:** PASSED 2026-11-04 15:55–16:01 PST (one session): 5/5 stations reached, plate ≥ 5.7 cm from the counter.
**Commit:** `74c082d` (movement, rebased onto main 0a3c7d7; not pushed at run time), deployed by `git archive` to
`~/chefbot_movement` (542 files, sha256 verified). Jack on the e-stop; arm driver off, joints locked
(q [0.137, -0.288, -0.192, -1.324, -0.009, 1.452, 0.659], from Redis: tip 1.15–1.40 m up, reach 0.548 m ahead).

### Setup
- Robot: `cd ~/chefbot_movement && CHEFBOT_PYTHON=~/chefbot_calib/env/bin/python chefbot/onboard/bringup/base.sh`
  (the teammate's env, for pyrealsense2). Driver: limits from onboard/config.yaml (0.125 m/s, 0.4 rad/s, accel 0.1 / 0.5),
  timeout 0.3 s, brake 0.25 m/s² / 0.8 rad/s².
- Mac: `real_nav go <station> --map maps/src_counter --margin 0.09 --speed 0.125`; after each, a watch-only
  pole RGB-D grab (scratch script `pole_snap.py`: refuses without live mocap; PNG + depth .npy + pose JSON).
- Margin 0.09, not HT-4d's 0.05: the footprint's base circle (0.33 m) is smaller than the plate's corners (0.369 m;
  teammate's review). Sim dry run first: `docs/evidence/counter/sim_station_lap_margin009.txt` (5/5).
- Link: Mac on 10.32.77.55, robot on SRC; RTT 88 ms average, 124 ms max, 28 ms jitter (40 pings), vs ~12 ms on 11-03.

### Observed
| Leg | Result | Final error | Peak speed (real_nav) |
|---|---|---|---|
| start → hug_a | first try held by the server's watchdog after ~5 cm (link stall); retry reached | 8.6 mm, 0.66° | 0.146 m/s |
| → counter_south | reached | 5.3 mm, 0.08° | 0.115 |
| → counter_east | **refused before moving**: "goal is in collision": its approach point (x 3.32) is 0.60 m from the east wall, the footprint with this arm pose + margin 0.653 m | — | — |
| → east_end (instead: same side, no straight-in approach, 0.82 m from the wall) | reached | 5.3 mm, 0.73° | **0.174** |
| → counter_north | reached | 5.1 mm, 0.22° | 0.097 |
| → counter_west | stopped by the laptop: no state for 0.55 s (link); base held | (0.69 m short) | 0.144 |
| → counter_west (retry) | reached | 6.1 mm, 0.24° | 0.104 |
| → hug_a | reached | 3.1 mm, 0.23° | 0.123 |

- 12.05 m driven, 3,078 laptop samples. Base centre ≥ **0.426 m** from the counter box (planned keep-out 0.42);
  **plate rectangle (19 × 22 in) ≥ 0.057 m** from it at every sample (the first check with the plate, not a circle).
- Every stop from a link stall was a hold (velocity 0, stack up); the driver's own timeout never had to fire.
- Pole images (Mac `runs/pole_snaps/20261104_lap/`, gitignored: people in frame): counter_south, east_end, counter_north,
  counter_west, hug_a, each taken at rest (moved < 0.03 mm during the grab), 1.3–1.9 s per grab (cold camera). Sharp.
  Depth valid only 20–34% of pixels (much of the scene beyond range; the arm fills the right third of every frame,
  too close for the D455).

### Limitations / open
- One session. Laptop-side poses only (no 100 Hz robot-side recorder this time).
- Peak speed 0.146 and 0.174 m/s against the 0.125 cap (real_nav's server-time estimate): with HT-4c's 0.298 lurch, the
  speed overshoot is still unexplained (review suggestion 2).
- counter_east is unreachable with this arm pose at margin 0.09 (approach point vs east wall); the footprint fix (review C1)
  separates the plate-vs-counter and arm-vs-wall cases, which should make the extra margin unnecessary.
- The cam_to_world in each JSON is the configured pole mount × base pose (tracker without the hand card in view).

---

## HT-5b: park for random counter work targets, baseline (straight-in) vs direct, after the heading fix

**Status:** PASSED 2026-11-04 16:35–16:47 PST (one session each). Robot `da85382` (driver heading fix), laptop `fa9138d`
(`real_nav park`). base.sh as HT-5a. Arm off, joints locked (footprint 0.556 m vs walls, 0.33 m vs the counter). Mac on
10.32.77.55 (not SRC), RTT ~9 ms.

Command (the same 8 targets, drawn by the pick task's start rule, seed 1):
`real_nav park --random 8 --seed 1 --map maps/src_counter --margin 0.09 --speed 0.125 --approach {0.25|0}` (yaw 0.3 rad/s,
so the driver's speed scale is 0.75: planar 0.094 m/s). Stances 0.53–0.56 m out, square to the side, target ahead.

### Heading fix (the HT-5a east_end failure)
Every move turned the short way: asked/taken -87/-87, -90/-90, -174/-174, +178/+179(asked), -178/-179, +90/+89 …
(15 parks, `turn_taken_deg` from the logs). Nothing like HT-5a's 285 deg wrong-way spin.

### Results (`docs/evidence/ht5b/park_metrics.txt`)
| | baseline (stop 0.25 m out, straight in) | direct (one smooth path) |
|---|---|---|
| parks | 7/7 (east skipped: its approach point is 0.49 m from the wall < 0.65) | **8/8** incl. east |
| like-for-like time, targets 2–6 | 190.9 s | **158.9 s (−17%)** |
| intermediate stops | 7 | **0** |
| within 5 cm → parked (median) | 5.70 s | **0.82 s** |
| final error | ≤ 12.5 mm, ≤ 0.45° | ≤ 8.3 mm, ≤ 1.30° |
| out from the edge / target lateral | 0.529–0.560 m / ≤ 4 mm | 0.526–0.558 m / ≤ 8 mm |
| plate rectangle → counter, min | 64 mm | 61 mm |
| base centre → walls, min | 1.00 m | 0.75 m |

Clearance against the measured stop (HT-4d: 78 mm from 0.131 m/s, last fresh command → rest incl. the 0.3 s timeout),
in the direction of travel: worst +60 mm (direct, at its closest the base moves along the counter at 0.007 m/s toward it);
projected to 0.125 m/s +46 mm. Not yet run at 0.125.

### Unexpected stops
Two holds (baseline target 3, direct target 6), both resumed. Direct target 6: mocap stopped updating on the robot
(age 0.10 s) about 0.1 s before the server's state stopped reaching the laptop (0.4 s); the server held on the laptop's
silence and then refused targets (mocap stale). Both mocap (Motive → robot) and the laptop link use the robot's Wi-Fi
(wlp88s0, power save already off, signal −63 dBm, quality 47/70): a robot-side Wi-Fi dropout. Executing the path on the
robot would not remove it (mocap still stalls).

### What limits speed (measured, direct runs)
- Cruise flat at ~0.072 m/s (median, every leg) against the 0.094 m/s scaled cap; turns ≤ 6 deg/s; nothing else stopping
  it. The follower sends a position target 0.12 m ahead (carrot) each tick and the driver's Ruckig plans to come to rest
  there: with 0.1 m/s² and ~0.1 s command lag the steady speed is ~0.07 m/s whatever the cap.
- 1.0 s of hold logging after every park (`go` tail_s), 0.5 s settle.

### Next
Faster (Jack: 2–3x, "a real kitchen"): a speed-dependent carrot (or velocity commands), higher driver velocity and
acceleration ceilings with stops re-measured at each step, speed limited by clearance near the counter from the measured
stop envelope, no hold tail. Closer to the counter (Jack: "a bit too far"): see current-state for the trade-off.
