# Maps: one folder per map, one format

Every map folder has the same files, read by one loader each, whatever built it:

| File | Read by | What |
|---|---|---|
| `grid.npy` | robot layer: `robot/nav/grid.py` `OccupancyGrid.load` | bool, `[row, col]`, row along +y, col along +x; true: occupied |
| `grid.yaml` | the same | `resolution` (m per cell), `origin` (world x, y of cell (0, 0)'s outer corner), `frame: world` |
| `heights.npy` | the same (`OccupancyGrid.heights`) | float, same shape, resolution and origin: each occupied cell's top above the floor (m); `inf` for walls and anything never to be reached over; 0 for free cells |
| `semantic_map.yaml` | agent: `WorldMap.load` (NAV_SPEC §7.1) | `frame: world`, `fixtures`, `objects`, rough locations and optional `facing`. A map with no semantics has a blank one: `{frame: world, fixtures: {}, objects: {}}` |
| `places.yaml` | `environments/real/build_map.py` `load_places` (optional) | named base poses, `{name: {x, y, yaw_deg[, approach_m]}}`: where the base goes, not things in the world (those are fixtures). Absent: no places |
| `spec.yaml` | `environments/real/build_map.py` (real maps only) | the hand-written description a real map is built from |

Obstacles are the furniture itself. The robot layer grows them by the robot's footprint at each heading and decides from
`heights` which parts of the robot pass over what (`robot/nav/footprint.py` `FootprintSpace.by_height`): a counter top
at 0.93 m blocks the base's plate but not an arm held higher; a wall blocks everything. That depends on the robot and
its arm pose, so it is not in the map.

## Builders (env-side, NAV_SPEC §11)

- `python -m chefbot.environments.robocasa.build_map --layout N --style M`: a RoboCasa scene, from its collision
  geometry (heights: the highest geometry over each cell in the height band; reaching the band's top, `inf`).
- `python -m chefbot.environments.real.build_map build maps/<name>/spec.yaml`: the real kitchen, from a spec (heights:
  each obstacle's `height_m`; none given, `inf`). An obstacle with `fixture: <name>` goes on the semantic map.

## The maps here

- `src_counter`: the SRC kitchen around the counter top (mocap frame). The counter's top is 0.93 m; its bounds are
  partly measured (see `spec.yaml`).
- `test_area`: an example open floor with one virtual box. Not measured.
- `robocasa_*`: their `heights.npy` were added with every obstacle `inf` (all walls, which is how they were planned on
  before heights existed), as RoboCasa was not installed where they were added. Rebuild them with the RoboCasa
  builder for real tops.
