# Future work: the sim agent's world model on the real robot

Written 2026-11-06, after the tomato demo's pieces (agents/real_kitchen, jobs/tomato_to_bowl.yaml).

## Where it stands

The real robot's agent (`agents/real_kitchen`) runs the same loop as the sim's (`agent/harnesses/plan_loop`: one tool
call per turn, the last 8 calls, a TASK STATE rebuilt every turn), but in a `chefbot-v1` session. Its skills are the
kitchen_use pick and `place_in/v1` (sandboxed episodes) and four `host-v1` skills written for the demo (`go_to`,
`find`, `approach`, `ready`; `scripts/real_kitchen.py`). The world it gets (`environments/real/agent_world.py`) has an
empty semantic map, no detector and a robot that moves only through its skills. So TASK STATE's PERCEPTION, SEARCHING
and KITCHEN sections are empty: its only memory is the window and the plan. `find` and `approach` stand in for
`detect` and `align`, and what `find` saw goes to the LLM as text, kept nowhere but for `approach`'s next call.

The sim agent (`agents/kitchen_planner`, robot-api-v1) has the full model: a semantic map (`agent/world/map.py`,
`semantic_map.yaml`: fixtures and objects with rough locations, `go_to` any name on it), what it has seen and done
(`nav.kitchen`, rendered as KITCHEN), the last `detect`'s ids in the world frame (`agent/world/detections.py`) with
their freshness (PERCEPTION), `nearby()`, SEARCHING, and `align(id)` for fine positioning.

## The work

Give the real robot a robot-api-v1 session, so kitchen_planner's own tools run on it:

1. **A real `Detector`** (`interface/detector.py`): the pole and wrist cameras through the vision server (Grounding
   DINO, SAM 2), each detection placed on the map by mocap: what `RealKitchen.find` does now.
2. **The robot layer's `Robot` on the real env as the RobotAPI**: what `real_nav` already drives (`make_robot`, the
   embodiment's navigator), with `align_to_point` parking by `goal.counter_stance`.
3. **A semantic map for `maps/src_kitchen_work`**: its fixtures are in the spec already (`fixture:` on each counter;
   `build_map` writes `semantic_map.yaml`), with `facing` toward the floor for each.
4. **pick_up and place as the real skills**: kitchen_use/run1_v155_fr3 and place_in/v1 run as sandboxed episodes
   today; in a robot-api-v1 session they'd be robot-api-v1 skills that run the episode on the real host.

Then KITCHEN, PERCEPTION, SEARCHING and `nearby` work on the robot as in sim, and the same agent runs in both.

## Check first: the arm's place on the base (likely 18 cm off)

`find` puts what a camera sees on the map through the camera's pose, which comes from the embodiment's arm mount:
`config/robot_tidybot_fr3.yaml` `arm_mount: xyz [0, 0, 0.335]`, the arm base right above the base pose. The map's
footprint (`interface/workspace.py` `Body.base_xy`, ASSUMED) says the base pose (mocap, odometry) is the plate's centre,
0.1825 m behind the arm, the plate 0.44 m up. If that is so, everything found is ~18 cm off toward the robot along its
heading (and the height 0.105 m), and `approach` parks that far off. The skills' own moves are not affected (they work
in one frame). Test: `python -m chefbot.scripts.real_kitchen find <object>` from two headings of the base turned in
place: a 45 deg turn moves the found position ~14 cm if the offset is real. The fix is the real robot's arm mount
(0.1825, 0, 0.44), with the pole camera's calibrated mount re-expressed in it. See also memory-map-next.md (gap 1).
