# Kitchen object map and go_to by name: state and next steps (2026-11-08)

Branch `kitchen-map` (pushed, no PR), from origin/main `c5fac1a`. Everything runs on the laptop. The robot runs only
`launch.sh` (drivers + onboard server), unchanged.

## What works (verified live, 2026-11-08)

| What | Result | Run (runs/real_kitchen/) |
|---|---|---|
| `explore south` (stove counter, then the island's south side, 4 stops) | 31–42 s per sweep incl. naming; every stop 0.2–0.6 cm off the stance | 20261108_122529_explore, 125145_session |
| `go_to <object>` (straight to the stance, `max` speed 0.25 m/s, 0.8 rad/s) | 33 go_tos reached; median 0.3 cm, worst 2.0 cm off | 122637_tour, 124354_tour, 125145_session |
| Jack's grading, random tour of 15 | right object 14/15, right spot 15/15 (the miss: the utensil holder, which overlaps the tongs standing in it) | 122637_tour |
| Gemini on the wrist picture's map spot | 8/8 on an unchanged layout; 7/10 right after a rearrange | 125145_session, 124354_tour |
| Removal: tomato moved stove counter → island | lost at its old spot → search re-swept south → found → go_to 0.0 cm off, pole and wrist both see it | 1235xx_session |
| Frames while driving (pole, 1 Hz) | drive-loop tick gap never over 0.1 s | all |
| Per-frame cost | frame transfer 0.18 s, YOLOE 0.19 s (Mac MPS, vision server), map 0.02 s | |

Jack has not graded the 8 cards of 125145_session or the 10 of 124354_tour yet (http://127.0.0.1:8780/grade).

## How to run it

```
# Mac, once: the SSH master (password); Claude rides it
ssh -M -S ~/.ssh/cm-chefbot -fN chefbot
ssh -S ~/.ssh/cm-chefbot -O forward -L 8765:localhost:8765 chefbot      # the onboard server's tunnel
# robot: Desk -> joints unlocked + FCI on (Desk via https://localhost:8443/desk/ after -L 8443:172.16.0.12:443)
bash chefbot/onboard/bringup/launch.sh                                      # in ~/ChefBot, sudo once
# Mac
~/.venvs/chefbot_vision/bin/python chefbot/runtime/vision_server.py --port 8791 --models yoloe
~/.venvs/chefbot_host/bin/python -m chefbot.scripts.live_view               # http://127.0.0.1:8780 (+ /grade)
set -a; . ../.env; set +a                                                   # GEMINI_API_KEY (outside the repo)
VISION_URL=http://127.0.0.1:8791 ~/.venvs/chefbot_host/bin/python -m chefbot.scripts.real_kitchen session \
    --route south --cameras pole wrist --capture both --naming end \
    --layout chefbot/perception/layouts/layout_2026-11-08.yaml [--objects <run>/objects_memory.json] [--search south]
kitchen> explore south | look | go_to tomato | tour 8 | objects | where | quit
```

## Where the code is

- `chefbot/scripts/real_kitchen.py`:
  - `RealKitchen.explore / look / go_to_object / check_at / tour / sweep_stops / stance_for`
  - `KitchenOptions` holds the switches: cameras, detector, capture, naming, layout, `wrist_vlm`, `search_route`
  - the CLI's `session` and `tour` commands
- `chefbot/perception/kitchen_map.py`: `KitchenMap` (MemoryMap with kept views, naming, save/load) and `Layout` (closed list, counts, sizes)
- `chefbot/perception/memory_map.py`: clusters, label votes, the gone-test (`_local_top`), the `reject` hook (the robot's own body)
- `chefbot/scripts/live_view.py`: `LiveFeed` writer, the one-screen page, `/grade`, `reviews.jsonl`
- `chefbot/runtime/vision_server.py`: `model: yoloe` (mask outlines)
- `maps/src_kitchen_work/sweep.yaml`: routes `full / south / stove`
- `chefbot/skills/explore/host_v1`; `go_to/host_v1` takes object names; `run_agent --objects` feeds the agent's KITCHEN map
- Tests:
  - `chefbot/tests/perception/test_kitchen_map.py` and `chefbot/tests/agents/test_real_kitchen_objects.py` (fake robot)
  - 3 `test_footprint` failures also fail on clean main on the Mac

Each run folder holds:
- `frames/<camera>/` (color jpg, depth png in mm, `frames.jsonl` with T_wc, K, base pose, detections)
- `objects.json`, `objects_memory.json` (with the views)
- `timings.jsonl`, `kitchen_log.jsonl`, `events.jsonl` (the review cards)

## Next coding session, in order

1. **Score the graded cards.** Pull `runs/live/reviews.jsonl` with `cards.json` from the live server. Report it per item and per run, with Gemini-wrist against Jack.
2. **Let the wrist answer correct the map.** When Gemini on the wrist says another item at the spot, or "other", relabel or demote the entry. Today's 3 misses were baking tray/tomato 5 cm apart, small bowl 2 vs the egg cracker, and the cumin bottle seen as "other".
3. **The ~5 cm height bias.** The stove counter reads 0.86 m against spec 0.81, and the stickers gave 0.80–0.82 on 11-06. Either the pole T's z or the `_arm_offset` plate height (0.44 vs 0.335) is off. Fit it from the saved frames: bare-counter depth points at known heights. `_local_top` hides it for the gone-test only.
4. **Duplicates in a busy map.** In `objects()`: frying pan ×3, ladle ×2, tongs vs the utensil holder. Options:
   - merge same-item entries within ~15 cm
   - raise `min_views` with drive capture
   - trust the layout count earlier, during the sweep, not only at export
5. **Pick start view.** Run a tour with the arm at READY (`ready`, needs Jack's "clear"), so the wrist frames are the pick skill's start. Check the object is in view and reach from `stance_for`'s stances.
6. **Data per layout.** A few layouts, one sweep each, with an answer-key photo (the 11-07 plan). The per-run `frames/` are already the dataset format.
7. **Open items:**
   - Named places still use `approach_m: 0.25` (teammate's `places.yaml`; out-then-in).
   - `inventory.txt` is superseded by layouts for tests.
   - Section 8 of the old `memory-map` runbook is superseded: it ran vision on the robot.

## Hardware notes from today

- **The base's "CAN Message is stale".** The motors drop out and the driver doesn't recover; relaunch `launch.sh`. Causes seen:
  - the wireless relay remote's batteries (replaced)
  - one `[base] interrupted` during a relaunch
  - Motor supply 12.50–12.53 V at start. It refused once with "Motor supply voltage is too low".
- **FCI off in Desk → "libfranka: Connection to FCI refused"** (port 172.16.0.12:1337 closed).
- **The arm held under OpenSai** all afternoon: 999.9 Hz, 0.0002° drift over 2 s, no sag. The sag incident is still not root-caused.
- **The pole camera missed its first frames after a restart.** One automatic retry now covers it.
