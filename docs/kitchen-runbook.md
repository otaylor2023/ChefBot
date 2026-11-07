# Kitchen runbook: driving and recording in the SRC kitchen

How to run a session next time, with the rules Jack set on 2026-11-05 and the traps found that day. Hardware results
are in [hardware-tests.md](hardware-tests.md) (HT-6 to HT-8).

## 0. Before anything
- **Arm:** locked, its driver off. The arm sagged once when unlocked (unresolved): nothing here unlocks or moves it.
- **E-stop:** a person holds the base's relay remote whenever the base can move.
- **Who has the robot:** `ssh chefbot 'ps -eo pid,etime,args | grep -E "driver.py|onboard.server|SampleClient|launch.sh|sai_franka|gamepad" | grep -v grep'`.
  One stack at a time. Don't stop the teammate's processes.
- **Network:** the robot should be on SRC (`nmcli -t -f NAME con show --active`). The Mac should be on SRC too: over
  Tailscale the round trip spiked to 474 ms on 2026-11-05 and the server's 0.3 s watchdog held the robot again and
  again. Driving from the robot (section 3) avoids that entirely.
- **SSH:** `ssh -fN -o ControlMaster=yes -o ControlPath=~/.ssh/cm-chefbot -o ControlPersist=8h chefbot` (password once).

## 1. Code on the robot
Each tree is a `git archive` of one commit into its own folder (the teammate's `~/ChefBot` is never touched):

| folder | what | used for |
|---|---|---|
| `~/chefbot_nav` | this branch (`kitchen-nav`) | `real_nav` run ON the robot, env `~/miniforge3/envs/chefbot-movement` (has mujoco, mink, coal, ompl) |
| `~/chefbot_movement` | PR #1 head | `base.sh` at main's 0.25 m/s cap |
| `~/chefbot_speedexp` | branch `speed-exp` (0.5 m/s cap, profiles exp35 / exp50) | `base.sh` for the faster runs (experiment, not main) |
| `~/chefbot_collect` | `collect_stream.py` (+ `collect_grab.py`) copied from `chefbot/onboard/` | the camera recorder, env `~/chefbot_calib/env` (pyrealsense2) |

Deploy: `git archive <commit> | ssh chefbot 'mkdir ~/X_new && tar -x -C ~/X_new'`, compare sha256 of every file with the
local archive, then swap folders (keep the old one).

## 2. Bring up the base (no motion)
```
ssh chefbot 'cd ~/chefbot_speedexp && (CHEFBOT_PYTHON=~/chefbot_calib/env/bin/python setsid nohup bash chefbot/onboard/bringup/base.sh > runs/base_sh.log 2>&1 < /dev/null &)'
```
(`~/chefbot_movement` instead for the 0.25 m/s cap.) Wait for `[onboard] serving on ws://127.0.0.1:8765`.
- **Stop it with `kill -TERM <base.sh pid>`.** A background-started script ignores SIGINT; base.sh traps TERM and stops
  the driver first (it brakes), then mocap and the server.
- Never `pkill -f <pattern>` over ssh when the pattern is in the ssh command itself: it kills your own shell.
- The server closes idle cameras only while a client is connected; a robot-side camera script needs the cameras free.

## 3. Drive: random A to B in the work area, from the robot
The robot runs `real_nav park` itself (no Wi-Fi in the drive loop); the Mac only starts it and reads the log:
```
ssh chefbot 'cd ~/chefbot_nav && printf "\n" | PYTHONPATH=$PWD setsid nohup ~/miniforge3/envs/chefbot-movement/bin/python -u \
  -m chefbot.scripts.real_nav park --random 20 --seed N \
  --mix kitchen_island_countertop:south 1-corner-countertop:north --x-range 0.3 1.55 --end-margin 0.05 \
  --map maps/src_kitchen --margin 0.05 --out 0.46 0.46 --approach 0 --profile exp50 --hold \
  --clearance 0.7 --turn-in-room --corridor 0.25 > runs/collect/park_NAME.log 2>&1 &'
```
- `--profile exp50` needs base.sh from `~/chefbot_speedexp` (cap 0.5); with main's cap use `--profile teleop` (0.25).
- `--hold`: stop chasing millimetres once inside 3 cm / 3 deg (ended the rocking at arrival).
- `--clearance 0.7 --turn-in-room`: paths down the middle of aisles and gaps, turns where there is room (both off by
  default in the code). `--clearance` is measured from the base centre, whose floor is ~0.42 m: below that it does nothing.
- Add `kitchen_island_countertop:north` to `--mix` to include trips around the island (through the fridge gap).
- Stops and retries: the run retries a target twice after a link stall or a brief mocap dropout; any other stop ends it.
- `--plan-only` checks stances with no motion.

## 4. Record (cameras) during the drive
```
ssh chefbot 'cd ~/chefbot_collect && (setsid nohup ~/chefbot_calib/env/bin/python collect_stream.py data/stream/NAME --hz 1 > data/stream_NAME.log 2>&1 < /dev/null & echo $! > /tmp/collect_stream.pid)'
... drive ...
ssh chefbot 'kill -TERM $(cat /tmp/collect_stream.pid)'
```
- Saves only while the base moves (+0.5 s either side), `--hz 1` frame per second; `--all` keeps every frame, `--hz 0` all 15 fps.
- Pole D455: color 1280x800, depth + IR left/right 1280x720. Wrist D405: color, depth, IR 640x480 (its USB 2 link).
  Per-stream intrinsics/extrinsics in `meta.json`; robot state (every mapped Motive body, odometry, commands, arm) in
  `state.jsonl`. Write a `NOTE.txt` per pass: layout, what moved (no apostrophes in shell-quoted notes).
- Size: ~250 MB per 20-trip pass at 1 Hz. **One pass per layout, then rearrange**; Jack's target for the data:
  "a rough idea of what the object is and where", not volume.
- Copy: `rsync -a --partial -e "ssh -o ControlPath=$HOME/.ssh/cm-chefbot" chefbot:chefbot_collect/data/stream/ robot_data/stream/`
  (~10 MB/s over the robot's Wi-Fi, which also carries mocap: don't copy big sets while driving).

## 5. Kitchen map rules (maps/src_kitchen, from Motive bodies)
- Everything is turned −2.96 deg from Motive's x axis (Motive's floor axes vs the room). The test counter is mapped at 0 deg.
- **Work area:** the U of aisles around the island; practice in the island–stove section (island south, stove counter
  north), x 0.3–1.55. The island's **sink** (its east 35 in, full depth) gets no targets.
- **No-go** (floor level): west of the island, its edge 20 in off the island's short end; the base's plate stays out,
  the arm may reach over it.
- **Fridge gap** (U's east aisle): open, but the base may only face along it (±20 deg): `heading_zones.yaml`.
- Stances 0.46 m out (plate ~18 cm from the edge), planning margin 0.05 m.
- The locked arm reaches ~0.68 m ahead: a 180 deg turn swings it east or west, so turns need room (handled by
  `--turn-in-room`; without it keep turning targets in x 1.0–1.5).

## 6. The pole camera was re-angled (2026-11-05): its calibration is stale
`environments/real/config.yaml` `cameras.pole.T` (camera in ChefBot's base frame) is the 2026-11-03 hand-card
calibration. Jack re-angled the pole camera before the 11-05 recordings, so world positions from the pole's depth
are off by that angle (and maybe height).
- **Fix (needs the teammate, it moves the arm):** `python -m chefbot.onboard.calibration.pole record ... --card` then
  `solve`; put the new `T` (and `track.T_flange_card`) in `environments/real/config.yaml`.
- **The recorded data stays usable:** every frame keeps raw depth, intrinsics and the base's mocap pose, so camera →
  world is recomputed offline with the new `T`: `world = planar(base pose) @ T @ point`.
- **Interim check without the arm:** fit the counter-top planes in the pole's depth while parked; their known heights
  (island 0.91, stove counter 0.83, from Motive) and edges give the camera's pitch, roll and height.

## 7. Known issues (2026-11-05)
- Pole D455 USB drops (14 in one 3-minute pass); the 10 Gb hub it shares logs `port 3: Cannot enable` (likely the wrist
  cable). Try the pole straight into a computer port: that may also bring back its IMU (reads 0 through the hub).
- Wrist D405 links at USB 2 (480 Mbps) in every port tried: 640x480 only.
- Planning a 180 deg trip across the island takes ~6 s on the Mac (pure-Python A* repairs); the south→north path still
  zig-zags.
- Arm model is ≥5 cm cautious at the fridge (gripper ~5 cm from the door where the model says contact).
- Map walls west (x −2.0) and north (y 4.32) are placeholders; L leg 2's depth is assumed (0.63 m).

## 8. Exploration sweep + live memory map (prepared 2026-11-07, not yet run on the robot)
Goal: one ~1-minute sweep (stove counter, then round the island) gives the inventory: what is on each counter and
roughly where, for `real_nav goto`. Route: `maps/src_kitchen/sweep.yaml` (7 stops; the stove counter's x 2.6 stop is
outside the 11-05 practised range). Vocabulary: `chefbot/perception/inventory.txt` (DRAFT: Jack edits it to the real
tools and ingredients). Offline replay of p2_layout3 through the live mapper: 84 frames at ~450 ms each on the M1's CPU
(never behind at 1 Hz), 27 entries, named by gemini-3.5-flash-lite in 12 s -> 24 objects.

**One-time setup on the robot** (a user venv, no system packages; ~1 GB):
```
ssh chefbot '~/miniforge3/bin/python3 -m venv ~/chefbot_vision && ~/chefbot_vision/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision && ~/chefbot_vision/bin/pip install ultralytics==8.4.204 pyyaml pillow'
scp ~/.cache/yoloe/yoloe-26s-seg.pt ~/.cache/yoloe/mobileclip2_b.ts chefbot:.cache/yoloe/
rsync -a --exclude .git --exclude runs mac_chefbot-movement/ chefbot:chefbot_sweep/      # this branch, beside ~/chefbot_nav
scp .env chefbot:chefbot_sweep/.env    # GEMINI_API_KEY for naming (or name on the Mac afterwards)
scp robot_data/calibration/environments_real_config_2026-11-05.yaml chefbot:chefbot_sweep/pole_cal_2026-11-05.yaml
```
Then time YOLOE on the robot's CPU on a recorded pass before relying on it live (the M1 CPU's ~450 ms is the reference).

**Per layout** (one sweep each, then rearrange):
1. Answer key first: a phone photo of each counter and `stream/NAME/ANSWER.txt`, one line per item: `counter: item`
   (island / stove / side). 30 s; this is what the map is scored against.
2. Base up (section 2), recorder on (section 4, `--hz 1`, data/stream/NAME).
3. Live map (robot):
```
ssh chefbot 'cd ~/chefbot_sweep && (PYTHONPATH=$PWD setsid nohup nice -n 10 ~/chefbot_vision/bin/python -u -m chefbot.perception.live_map \
  ~/chefbot_collect/data/stream/NAME --config pole_cal_2026-11-05.yaml --threads 4 --out maps/src_kitchen/objects.json \
  --name gemini-3.5-flash-lite --env-file .env > runs/live_NAME.log 2>&1 < /dev/null &)'
```
   (`nice` + `--threads 4`: the detector must not starve the drive loop on the same CPU; watch for holds in the park log.)
4. Sweep: `--plan-only` first, then for real (Jack's "clear"):
```
... -m chefbot.scripts.real_nav park --sweep maps/src_kitchen/sweep.yaml --map maps/src_kitchen --end-margin 0.05 \
  --margin 0.05 --out 0.46 0.46 --approach 0 --profile exp50 --hold --clearance 0.7 --turn-in-room --corridor 0.25
```
5. Stop the recorder (live_map finishes when the recording's summary.json appears: drops one-view entries, merges,
   names). Then `real_nav goto NAME --map maps/src_kitchen --plan-only` for three items, then drive them.
