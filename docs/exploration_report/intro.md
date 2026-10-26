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
