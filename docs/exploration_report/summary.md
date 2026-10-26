# ChefBot exploration: summary

## What was built

- **An exploration task.** The robot starts from a map of the kitchen's fixtures only (counters,
  appliances, and optionally every cabinet and drawer, each with its height). It must find every
  loose object and write `explored_map.yaml`. A scorer compares that with the true map, which is
  built from the simulator only after the run; the robot never sees it.
- **A scripted explorer route,** plus an LLM explorer with a prompt:
  - visit every fixture, and look behind every door: fridge, microwave, dishwasher, oven, cabinets,
    drawers
  - close each door again
  - stand at the left and right ends of long counters and islands
- **Three detectors:**
  - **exact:** the simulator's detector, perfect perception, as a reference
  - **VLM:** a vision language model that names and boxes objects in the camera picture
  - **OWLv2:** a free open-vocabulary detector

  For the two camera detectors, the 3D position comes from the depth image under each box.
- **Fixes along the way:**
  - door handling: ajar doors, closing a door by its name
  - long counters
  - duplicate map entries
  - a camera the arm does not block (the fridge view)
  - a VLM spending cap
- **Position accuracy:** a report and tests that measure each object's error against the truth.
- **Root cause analysis:** each run is replayed exactly to see, for every object in every picture,
  why it was missed or misnamed.
- **Improvements tried:**
  - a name vote per spot
  - zooming on small boxes
  - driving closer to small objects
  - hybrid naming: OWLv2 boxes, with SigLIP 2 naming the small ones
  - near-synonyms as one name (cup and mug, pan and saucepan, lemon and lime)
  - SAM 3 as a checker
  - SAM 2.1 masks for depth

## Headline numbers

Objects found. Names are compared with near-synonyms as one name (cup and mug, pan and
saucepan, lemon and lime, bottled water and water bottle), as in all tables below.

| | exact detector | VLM | OWLv2 (baseline) | best free setup (OWLv2 + vote + hybrid + aliases) |
|---|---|---|---|---|
| layout 1 (6 objects) | 6/6 | 6/6 | 5/6 | not run |
| layout 7 (14 objects) | 13/14 | 10/14 (earlier route) | 9/14 | **10/14** |
| layout 25 (14 objects) | 12/14 | 7/14 (earlier route) | 6/14 | not run |
| layout 39 (14 objects) | 12/14 | 8/14 (earlier route) | 6/14 | not run |
| position error, median (all found objects) | 0.1 cm (the pan's reference point aside) | about 4 cm | about 4 to 6 cm | about 3 cm |
| naming, offline on 758 saved sightings of real objects | | 78% right (86% with aliases) | 43% | 63% (SAM 3 checker: 71% on a sample) |

- **SAM 3 as a checker on layout 7** (on top of the best free setup): 9/14 with the strict version
  (it drops boxes no mask fits) and 10/14 with the gentle one (it keeps them). Both runs have their
  own section below.
- **What limits each detector:**
  - **exact:** only what the route cannot reach (drawers too low for the arm, one cabinet it cannot
    open, one microwave out of reach)
  - **camera detectors:** mostly naming, and objects that cover too few pixels (under about 300
    px they are rarely found; over 3000 px both keep about 85%)
- **Positions:** a typical error is a few cm, from turning a box into a point with depth. It does
  not shrink much with distance.
- **The VLM runs on layouts 7, 25 and 39** used the earlier route: no cabinets or drawers opened,
  and the arm blocked the fridge. No VLM run was made on the current route, to save cost.

## Layouts explored

| layout | map fixtures | cabinets and drawers | objects | notes |
|---|---|---|---|---|
| 1 | 9 | not on the map (default task) | 6: egg in the fridge; mug, bowl, bottled water, pan, cutting board on counters | the default kitchen |
| 7 | 18 (9 counters incl. an island) | 48 (2 above the camera, skipped) | 14 (hard task) | U shape with an island |
| 25 | 16 (8 counters) | 65 (10 above the camera) | 14 | the most cabinets |
| 39 | 12 (oven and toaster oven) | 51 (1 above the camera) | 14 | |

The hard task's 14 objects:

- **in the fridge:** egg, milk
- **in the microwave:** bowl
- **on counters:** mug, cup, apple, lemon, banana, spoon, bottled water, pan, cutting board
- **hidden:** a plate in a cabinet, a fork in a drawer

## How to read the runs below

- **Each run's table** lists every true object, whether it was found, the name and fixture it got,
  and its xy and 3D error. Errors over 15 cm (the scorer's tolerance) are in bold.
- **"Wrong name"** means an object of another name was mapped on that spot (within 15 cm).
- **False finds** are objects mapped where there is no true object. Some are real things the true
  map leaves out (knives in a knife block, decorations).
- **The map** shows true positions (dots) and found positions (x), joined by an error line.
- **The video** replays every picture the detector took:
  - true objects in green (found and named right), orange (wrong name) or red (missed)
  - the detector's own boxes and names in blue
  - a caption naming each mistake
- **The gallery** shows, for each missed, misnamed or more-than-10-cm-off object, the picture where
  it was best visible, with a zoomed crop, its size in pixels, its distance and the cause.


## Layout pictures

Each kitchen as the runs had it (same task and seed): seen from above with fixture names, the objects (numbered, listed on the right) and the robot's start; the plan from the start map; and two angled views. Wall cabinets and walls are left out from above, and the near walls in the angled views. Counter colors can differ from the runs' pictures: RoboCasa picks textures at random.

- Layout 1: [from above, labeled](media/layouts/layout001_top.png), [plan from the start map, with cabinets and drawers](media/layouts/layout001_plan.png), [angled view](media/layouts/layout001_view1.png), [angled view from the other side](media/layouts/layout001_view2.png)
- Layout 7: [from above, labeled](media/layouts/layout007_top.png), [plan from the start map, with cabinets and drawers](media/layouts/layout007_plan.png), [angled view](media/layouts/layout007_view1.png), [angled view from the other side](media/layouts/layout007_view2.png)
- Layout 25: [from above, labeled](media/layouts/layout025_top.png), [plan from the start map, with cabinets and drawers](media/layouts/layout025_plan.png), [angled view](media/layouts/layout025_view1.png), [angled view from the other side](media/layouts/layout025_view2.png)
- Layout 39: [from above, labeled](media/layouts/layout039_top.png), [plan from the start map, with cabinets and drawers](media/layouts/layout039_plan.png), [angled view](media/layouts/layout039_view1.png), [angled view from the other side](media/layouts/layout039_view2.png)


### Layout 1, exact detector

Run `2026-10-25_145153_explore_robocasa_scripted_s0`: 6/6 found, 0 false finds.

| true object | true fixture | found | name given | fixture given | xy error (cm) | 3D error (cm) |
|---|---|---|---|---|---|---|
| bottled water | counter_1 | yes | bottled water | counter_1 | 0.1 | 0.1 |
| bowl | counter_1 | yes | bowl | counter_1 | 0.1 | 0.1 |
| cutting board | counter_1 | yes | cutting board | counter_1 | 0.1 | 0.1 |
| egg | fridge | yes | egg | fridge | 0.1 | 0.0 |
| mug | counter_1 | yes | mug | counter_1 | 0.0 | 0.0 |
| pan | counter_1 | yes | pan | counter_1 | 10.7 | 10.7 |

### Layout 1, VLM

Run `2026-10-19_184125_explore_robocasa_scripted_s0`: 6/6 found, 2 false finds.

| true object | true fixture | found | name given | fixture given | xy error (cm) | 3D error (cm) |
|---|---|---|---|---|---|---|
| bottled water | counter_1 | yes | bottled water | counter_1 | 2.1 | 2.3 |
| bowl | counter_1 | yes | bowl | counter_1 | 11.5 | 12.4 |
| cutting board | counter_1 | yes | cutting board | counter_1 | 4.8 | 5.2 |
| egg | fridge | yes | egg | fridge | 1.9 | 2.2 |
| mug | counter_1 | yes | mug | counter_1 | 3.5 | 4.3 |
| pan | counter_1 | yes | pan | counter_1 | 12.9 | 12.9 |

False finds: knife_1 at (3.60, -0.28) in counter_2, knife_2 at (3.65, -0.26) in counter_2.

### Layout 1, OWLv2

Run `2026-10-25_145221_explore_robocasa_scripted_s0`: 5/6 found, 5 false finds.

| true object | true fixture | found | name given | fixture given | xy error (cm) | 3D error (cm) |
|---|---|---|---|---|---|---|
| bottled water | counter_1 | yes | bottled water | counter_1 | 2.1 | 4.2 |
| bowl | counter_1 | yes | bowl | counter_1 | 5.9 | 6.2 |
| cutting board | counter_1 | yes | cutting board | counter_1 | 6.1 | 6.2 |
| egg | fridge | no |  |  |  |  |
| mug | counter_1 | yes | cup | counter_1 | 4.5 | 4.9 |
| pan | counter_1 | yes | pan | dishwasher | 9.6 | 9.7 |

False finds: bar soap at (0.60, -0.26) in coffee machine, blender jug at (2.01, -0.20) in counter_1, cheese grater at (3.63, -0.25) in counter_2, knife at (3.67, -0.27) in counter_2, soap dispenser at (1.99, -0.21) in counter_1.

### Layout 7, exact detector (current route)

Run `2026-10-25_145308_explore_robocasa_scripted_s0_layout007`: 13/14 found, 0 false finds.

| true object | true fixture | found | name given | fixture given | xy error (cm) | 3D error (cm) |
|---|---|---|---|---|---|---|
| apple | counter_1 | yes | apple | counter_1 | 0.1 | 0.0 |
| banana | counter_1 | yes | banana | counter_1 | 0.0 | 0.2 |
| bottled water | counter_1 | yes | bottled water | counter_1 | 0.0 | 0.1 |
| bowl | microwave | yes | bowl | microwave | 0.1 | 0.1 |
| cup | counter_1 | yes | cup | counter_1 | 0.0 | 0.0 |
| cutting board | counter_1 | yes | cutting board | counter_1 | 0.1 | 0.1 |
| egg | fridge | yes | egg | fridge | 0.1 | 0.0 |
| fork | drawer_4 | no |  |  |  |  |
| lemon | counter_1 | yes | lemon | counter_1 | 0.1 | 0.2 |
| milk | fridge | yes | milk | fridge | 0.1 | 0.1 |
| mug | counter_1 | yes | mug | counter_1 | 0.0 | 0.0 |
| pan | counter_1 | yes | pan | counter_1 | 10.7 | 10.7 |
| plate | cabinet_3 | yes | plate | cabinet_3 | 0.0 | 0.1 |
| spoon | counter_1 | yes | spoon | counter_1 | 0.0 | 0.4 |

### Layout 7, VLM (earlier route)

Run `2026-10-20_144100_explore_robocasa_scripted_s0_layout007`: 10/14 found, 8 false finds.

| true object | true fixture | found | name given | fixture given | xy error (cm) | 3D error (cm) |
|---|---|---|---|---|---|---|
| apple | counter_1 | yes | apple | counter_1 | 2.0 | 2.1 |
| banana | counter_1 | yes | banana | counter_1 | 12.5 | 12.7 |
| bottled water | counter_1 | yes | bottled water | counter_1 | 2.0 | 2.3 |
| bowl | microwave | yes | bowl_1 | microwave | 10.6 | 10.8 |
| cup | counter_1 | yes | cup | counter_1 | 3.8 |  |
| cutting board | counter_1 | yes | cutting board | counter_1 | 6.5 | 5.9 |
| egg | fridge | no |  |  |  |  |
| fork | microwave | no |  |  |  |  |
| lemon | counter_1 | yes | lemon | counter_1 | 2.0 | 2.7 |
| milk | fridge | no |  |  |  |  |
| mug | counter_1 | yes | mug | counter_1 | 4.2 | 4.1 |
| pan | counter_1 | yes | pan | counter_1 | 8.2 | 8.3 |
| plate | counter_4 | no |  |  |  |  |
| spoon | counter_1 | yes | spoon | counter_1 | 1.7 | 1.7 |

False finds: bowl_2 at (3.39, -2.76) in counter_5, fork at (6.38, -1.94) in counter_8, jug at (1.33, -0.17) in counter_4, knife at (0.69, -1.13) in counter_2, paper towel holder at (1.35, -0.17) in counter_4, spatula at (6.38, -1.85) in counter_8, whisk at (6.39, -1.94) in counter_8, wooden spoon at (6.39, -2.04) in counter_8.

### Layout 7, OWLv2 baseline (current route)

Run `2026-10-25_175806_explore_robocasa_scripted_s0_layout007`: 9/14 found, 19 false finds.

| true object | true fixture | found | name given | fixture given | xy error (cm) | 3D error (cm) |
|---|---|---|---|---|---|---|
| apple | counter_1 | yes | apple | counter_1 | 2.0 | 2.2 |
| banana | counter_1 | yes | banana | counter_1 | 2.0 | 2.5 |
| bottled water | counter_1 | yes | bottled water_1 | counter_1 | 2.1 | 3.0 |
| bowl | microwave | yes | bowl | microwave | 10.8 | 11.2 |
| cup | counter_1 | yes | cup | counter_1 | 3.0 | 3.6 |
| cutting board | counter_1 | yes | cutting board_1 | counter_1 | 6.6 | 6.1 |
| egg | fridge | wrong name | bar soap_1 | fridge | 1.9 |  |
| fork | drawer_4 | no |  |  |  |  |
| lemon | counter_1 | wrong name | pear | counter_1 | 2.4 |  |
| milk | fridge | wrong name | bottled water_2 | fridge | 2.5 |  |
| mug | counter_1 | yes | mug | counter_1 | 3.6 | 4.0 |
| pan | counter_1 | yes | pan | counter_2 | 9.5 | 9.6 |
| plate | cabinet_3 | no |  |  |  |  |
| spoon | counter_1 | yes | spoon | counter_1 | 0.9 | 0.8 |

False finds: bar at (0.85, -0.69) in counter_2, bar soap_2 at (6.10, -0.89) in coffee machine, candle at (1.39, -0.15) in counter_4, condiment bottle at (0.48, -1.41) in counter_1, cutting board_2 at (0.39, -1.16) in counter_1, fork at (6.39, -1.94) in counter_8, ladle_1 at (0.47, -1.60) in counter_1, ladle_2 at (6.38, -2.04) in counter_8, oven tray at (5.96, -0.91) in coffee machine, pickle at (0.52, -2.00) in counter_1, reamer_1 at (0.58, -1.15) in counter_1, reamer_2 at (1.36, -0.17) in counter_4, soap dispenser_1 at (1.36, -0.17) in counter_4, soap dispenser_2 at (5.59, -2.90) in fridge, spatula_1 at (0.46, -1.63) in counter_1, spatula_2 at (0.57, -1.15) in counter_1, spray at (0.48, -1.40) in counter_1, whisk at (6.39, -1.94) in counter_8, wooden spoon at (6.39, -2.04) in counter_8.

### Layout 7, best free setup: OWLv2 + vote + hybrid + aliases (current route)

Run `2026-10-26_110513_explore_robocasa_scripted_s0_layout007`: 10/14 found, 10 false finds.

| true object | true fixture | found | name given | fixture given | xy error (cm) | 3D error (cm) |
|---|---|---|---|---|---|---|
| apple | counter_1 | yes | apple | counter_1 | 2.0 | 2.1 |
| banana | counter_1 | yes | banana_1 | counter_1 | 1.6 | 2.3 |
| bottled water | counter_1 | yes | bottled water_1 | counter_1 | 1.5 | 2.4 |
| bowl | microwave | yes | bowl | microwave | 10.8 | 11.2 |
| cup | counter_1 | yes | mug_2 | counter_1 | 3.4 | 3.8 |
| cutting board | counter_1 | yes | cutting board | counter_1 | 6.4 | 6.5 |
| egg | fridge | yes | egg | fridge | 1.9 | 2.1 |
| fork | drawer_4 | no |  |  |  |  |
| lemon | counter_1 | yes | lemon | counter_1 | 2.3 | 2.7 |
| milk | fridge | wrong name | bottled water_2 | fridge | 2.5 |  |
| mug | counter_1 | yes | mug_1 | counter_1 | 3.4 | 4.1 |
| pan | counter_1 | yes | pan | counter_2 | 13.6 | 13.7 |
| plate | cabinet_3 | no |  |  |  |  |
| spoon | counter_1 | wrong name | banana_2 | counter_1 | 10.0 |  |

False finds: bar at (0.85, -0.69) in counter_2, ladle at (0.35, -0.93) in counter_1, oven tray at (5.96, -0.91) in coffee machine, pickle at (0.48, -1.88) in counter_1, soap dispenser_1 at (1.36, -0.17) in counter_4, soap dispenser_2 at (5.59, -2.90) in fridge, spatula_1 at (6.39, -2.08) in counter_8, spatula_2 at (6.39, -1.97) in counter_8, spatula_3 at (6.39, -1.85) in counter_8, sugar cube at (6.10, -0.89) in coffee machine.

### Layout 7, best free setup + SAM 3 checker, gentle (current route)

Run `2026-10-26_121527_explore_robocasa_scripted_s0_layout007`: 10/14 found, 11 false finds.

| true object | true fixture | found | name given | fixture given | xy error (cm) | 3D error (cm) |
|---|---|---|---|---|---|---|
| apple | counter_1 | wrong name | cherry | counter_1 | 2.2 |  |
| banana | counter_1 | yes | banana | counter_1 | 1.6 | 2.1 |
| bottled water | counter_1 | yes | bottled water_1 | counter_1 | 1.5 | 2.4 |
| bowl | microwave | yes | bowl | microwave | 10.8 | 11.2 |
| cup | counter_1 | yes | mug_2 | counter_1 | 3.4 | 3.8 |
| cutting board | counter_1 | yes | cutting board | counter_1 | 6.3 | 6.5 |
| egg | fridge | yes | egg | fridge | 1.9 | 2.1 |
| fork | drawer_4 | no |  |  |  |  |
| lemon | counter_1 | yes | lemon_2 | counter_1 | 1.9 | 2.6 |
| milk | fridge | wrong name | bottled water_2 | fridge | 2.5 |  |
| mug | counter_1 | yes | mug_1 | counter_1 | 3.4 | 4.1 |
| pan | counter_1 | yes | pan | counter_2 | 13.1 | 13.0 |
| plate | cabinet_3 | no |  |  |  |  |
| spoon | counter_1 | yes | spoon_1 | counter_1 | 0.9 | 0.5 |

False finds: bar at (0.85, -0.69) in counter_2, ladle at (0.35, -0.93) in counter_1, lemon_1 at (0.51, -2.01) in counter_1, oven tray at (5.96, -0.91) in coffee machine, reamer at (1.36, -0.17) in counter_4, soap dispenser at (5.59, -2.90) in fridge, spatula_1 at (6.39, -1.95) in counter_8, spatula_2 at (6.39, -1.85) in counter_8, spoon_2 at (0.47, -1.66) in counter_1, sugar cube at (6.10, -0.89) in coffee machine, wooden spoon at (6.38, -2.04) in counter_8.

### Layout 7, best free setup + SAM 3 checker, strict (current route; table and map only)

Run `2026-10-26_115859_explore_robocasa_scripted_s0_layout007`: 9/14 found, 6 false finds.

| true object | true fixture | found | name given | fixture given | xy error (cm) | 3D error (cm) |
|---|---|---|---|---|---|---|
| apple | counter_1 | wrong name | cherry | counter_1 | 2.2 |  |
| banana | counter_1 | yes | banana | counter_1 | 1.6 | 2.1 |
| bottled water | counter_1 | yes | bottled water_1 | counter_1 | 1.5 | 2.4 |
| bowl | microwave | yes | bowl | microwave | 10.8 | 11.2 |
| cup | counter_1 | yes | mug_2 | counter_1 | 3.4 | 3.8 |
| cutting board | counter_1 | yes | cutting board | counter_1 | 6.3 | 6.5 |
| egg | fridge | yes | egg | fridge | 1.9 | 2.1 |
| fork | drawer_4 | no |  |  |  |  |
| lemon | counter_1 | no |  |  |  |  |
| milk | fridge | wrong name | bottled water_2 | fridge | 2.5 |  |
| mug | counter_1 | yes | mug_1 | counter_1 | 3.4 | 4.1 |
| pan | counter_1 | yes | pan | counter_2 | 13.1 | 13.0 |
| plate | cabinet_3 | no |  |  |  |  |
| spoon | counter_1 | yes | spoon_1 | counter_1 | 0.9 | 0.5 |

False finds: oven tray at (5.95, -0.92) in coffee machine, soap dispenser at (1.35, -0.17) in counter_4, spatula_1 at (6.39, -1.95) in counter_8, spatula_2 at (6.39, -1.85) in counter_8, spoon_2 at (0.47, -1.66) in counter_1, wooden spoon at (6.38, -2.04) in counter_8.

### Layout 25, exact detector (current route)

Run `2026-10-25_150237_explore_robocasa_scripted_s0_layout025`: 12/14 found, 0 false finds.

| true object | true fixture | found | name given | fixture given | xy error (cm) | 3D error (cm) |
|---|---|---|---|---|---|---|
| apple | counter_1 | yes | apple | counter_1 | 0.0 | 0.0 |
| banana | counter_1 | yes | banana | counter_1 | 0.4 | 1.2 |
| bottled water | counter_1 | yes | bottled water | counter_1 | 0.0 | 0.1 |
| bowl | microwave | no |  |  |  |  |
| cup | counter_1 | yes | cup | counter_1 | 0.0 | 0.0 |
| cutting board | counter_1 | yes | cutting board | counter_1 | 0.0 | 0.1 |
| egg | fridge | yes | egg | fridge | 0.1 | 0.0 |
| fork | drawer_1 | no |  |  |  |  |
| lemon | counter_1 | yes | lemon | counter_1 | 0.1 | 0.1 |
| milk | fridge | yes | milk | fridge | 0.0 | 0.1 |
| mug | counter_1 | yes | mug | counter_1 | 0.0 | 0.0 |
| pan | counter_1 | yes | pan | counter_1 | 10.7 | 10.7 |
| plate | cabinet_24 | yes | plate | cabinet_24 | 0.1 | 0.0 |
| spoon | counter_1 | yes | spoon | counter_1 | 0.1 | 0.5 |

### Layout 25, VLM (earlier route)

Run `2026-10-20_144336_explore_robocasa_scripted_s0_layout025`: 7/14 found, 1 false finds.

| true object | true fixture | found | name given | fixture given | xy error (cm) | 3D error (cm) |
|---|---|---|---|---|---|---|
| apple | counter_1 | wrong name | tomato | counter_1 | 2.0 |  |
| banana | counter_1 | yes | banana | counter_1 | 4.5 | 5.1 |
| bottled water | counter_1 | yes | bottled water_1 | counter_1 | 1.8 | 2.0 |
| bowl | microwave | no |  |  |  |  |
| cup | counter_1 | yes | mug | counter_1 | 10.8 | 11.1 |
| cutting board | counter_1 | yes | cutting board | counter_1 | 6.0 | 6.1 |
| egg | fridge | no |  |  |  |  |
| fork | counter_1 | no |  |  |  |  |
| lemon | counter_1 | yes | lemon | counter_1 | 2.2 | 2.4 |
| milk | fridge | wrong name | bottled water_2 | fridge | 2.4 |  |
| mug | counter_1 | no |  |  |  |  |
| pan | counter_1 | yes | pan | counter_1 | 13.1 | 13.1 |
| plate | counter_3 | no |  |  |  |  |
| spoon | counter_1 | yes | spoon | counter_1 | 0.5 | 0.5 |

False finds: avocado at (0.48, -4.28) in counter_1.

### Layout 25, OWLv2 baseline (current route)

Run `2026-10-25_181121_explore_robocasa_scripted_s0_layout025`: 6/14 found, 9 false finds.

| true object | true fixture | found | name given | fixture given | xy error (cm) | 3D error (cm) |
|---|---|---|---|---|---|---|
| apple | counter_1 | yes | apple | counter_1 | 2.5 | 2.5 |
| banana | counter_1 | yes | banana | counter_1 | 4.8 | 5.3 |
| bottled water | counter_1 | yes | bottled water_1 | counter_1 | 1.8 | 2.0 |
| bowl | microwave | no |  |  |  |  |
| cup | counter_1 | wrong name | measuring cup | counter_1 | 2.9 |  |
| cutting board | counter_1 | yes | cutting board | counter_1 | 6.0 | 6.4 |
| egg | fridge | wrong name | bar soap_2 | fridge | 1.9 |  |
| fork | drawer_1 | no |  |  |  |  |
| lemon | counter_1 | wrong name | sponge | counter_1 | 1.9 |  |
| milk | fridge | wrong name | bottled water_2 | fridge | 2.3 |  |
| mug | counter_1 | yes | cup | counter_1 | 4.7 | 5.2 |
| pan | counter_1 | yes | pan | counter_1 | 7.5 | 7.9 |
| plate | cabinet_24 | no |  |  |  |  |
| spoon | counter_1 | wrong name | fork | counter_1 | 0.6 |  |

False finds: bar_1 at (2.65, -0.73) in burner_5, bar_2 at (4.32, -1.70) in fridge, bar soap_1 at (0.28, -1.68) in cabinet_8, blender jug at (0.28, -4.06) in counter_1, condiment bottle at (0.47, -3.66) in counter_1, reamer at (0.48, -3.67) in counter_1, saucepan lid at (0.48, -4.46) in counter_1, soap dispenser at (0.30, -4.08) in counter_1, spray at (0.47, -3.66) in counter_1.

### Layout 39, exact detector (current route)

Run `2026-10-25_151243_explore_robocasa_scripted_s0_layout039`: 12/14 found, 0 false finds.

| true object | true fixture | found | name given | fixture given | xy error (cm) | 3D error (cm) |
|---|---|---|---|---|---|---|
| apple | counter_1 | yes | apple | counter_1 | 0.0 | 0.0 |
| banana | counter_1 | yes | banana | counter_1 | 0.0 | 0.0 |
| bottled water | counter_1 | yes | bottled water | counter_1 | 0.0 | 0.1 |
| bowl | microwave | yes | bowl | microwave | 0.0 | 0.1 |
| cup | counter_1 | yes | cup | counter_1 | 0.0 | 0.0 |
| cutting board | counter_1 | yes | cutting board | counter_1 | 0.1 | 0.1 |
| egg | fridge | yes | egg | fridge | 0.1 | 0.0 |
| fork | drawer_3 | no |  |  |  |  |
| lemon | counter_1 | yes | lemon | counter_1 | 0.1 | 0.2 |
| milk | fridge | yes | milk | fridge | 0.1 | 0.1 |
| mug | counter_1 | yes | mug | counter_1 | 0.0 | 0.0 |
| pan | counter_1 | yes | pan | counter_1 | 10.7 | 10.7 |
| plate | cabinet_18 | no |  |  |  |  |
| spoon | counter_1 | yes | spoon | counter_1 | 0.1 | 0.1 |

### Layout 39, VLM (earlier route)

Run `2026-10-20_144626_explore_robocasa_scripted_s0_layout039`: 8/14 found, 2 false finds.

| true object | true fixture | found | name given | fixture given | xy error (cm) | 3D error (cm) |
|---|---|---|---|---|---|---|
| apple | counter_1 | yes | apple | counter_1 | 2.6 | 3.0 |
| banana | counter_1 | yes | banana | counter_1 | 2.2 | 2.5 |
| bottled water | counter_1 | yes | bottled water | counter_1 | 1.9 | 2.2 |
| bowl | microwave | wrong name | bread | microwave | 11.8 |  |
| cup | counter_1 | yes | mug | counter_1 | **16.3** | **16.7** |
| cutting board | counter_1 | yes | cutting board | counter_1 | 5.4 | 5.5 |
| egg | fridge | no |  |  |  |  |
| fork | counter_1 | no |  |  |  |  |
| lemon | counter_1 | yes | lemon | counter_1 | 2.3 | 2.8 |
| milk | fridge | wrong name | bottled drink | fridge | 2.6 |  |
| mug | counter_1 | yes | cup | counter_1 | 3.4 | 6.0 |
| pan | counter_1 | no |  |  |  |  |
| plate | fridge | no |  |  |  |  |
| spoon | counter_1 | yes | spoon | counter_1 | 0.0 | 0.3 |

False finds: knife at (2.62, -0.27) in counter_2, lime at (2.54, -5.84) in counter_1.

### Layout 39, OWLv2 baseline (current route)

Run `2026-10-25_182428_explore_robocasa_scripted_s0_layout039`: 6/14 found, 20 false finds.

| true object | true fixture | found | name given | fixture given | xy error (cm) | 3D error (cm) |
|---|---|---|---|---|---|---|
| apple | counter_1 | yes | apple | counter_1 | 3.3 | 3.0 |
| banana | counter_1 | yes | banana | counter_1 | 1.9 | 2.4 |
| bottled water | counter_1 | yes | bottled water_1 | counter_1 | 1.6 | 2.4 |
| bowl | microwave | wrong name | cutting board_2 | microwave | 11.2 |  |
| cup | counter_1 | yes | mug | counter_1 | 3.3 | 4.1 |
| cutting board | counter_1 | yes | cutting board_1 | counter_1 | 5.2 | 5.3 |
| egg | fridge | wrong name | bar soap | fridge | 1.9 |  |
| fork | drawer_3 | no |  |  |  |  |
| lemon | counter_1 | wrong name | avocado | counter_1 | 2.1 |  |
| milk | fridge | wrong name | bottled water_2 | fridge | 2.6 |  |
| mug | counter_1 | wrong name | straw | counter_1 | 5.0 |  |
| pan | counter_1 | yes | pan | counter_1 | 8.9 | 9.0 |
| plate | cabinet_18 | no |  |  |  |  |
| spoon | counter_1 | wrong name | scissors | counter_1 | 0.1 |  |

False finds: bar at (3.62, -5.63) in microwave, can opener at (2.71, -6.01) in counter_1, candle at (1.52, -0.15) in counter_2, cheese grater at (2.65, -0.25) in counter_2, condiment bottle at (1.99, -5.87) in counter_1, dish brush at (2.64, -0.25) in counter_2, kiwi at (2.73, -5.84) in counter_1, knife_1 at (1.24, -5.90) in counter_1, knife_2 at (2.15, -5.88) in counter_1, knife_3 at (2.67, -0.27) in counter_2, ladle at (1.68, -5.77) in counter_1, pear at (2.56, -5.84) in counter_1, pizza cutter at (2.15, -5.88) in counter_1, reamer_1 at (1.98, -5.87) in counter_1, reamer_2 at (2.64, -0.25) in counter_2, soap dispenser at (1.49, -0.16) in counter_2, spatula at (1.68, -5.77) in counter_1, sponge_1 at (2.55, -5.84) in counter_1, sponge_2 at (2.70, -0.10) in counter_2, strainer at (1.72, -5.77) in counter_1.
