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
  - a close look with the wrist camera at uncertain objects, and moving an object that hides another
    (see "Close looks" below)

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
- **Close looks with the wrist camera** (on top of the best free setup): layout 7 11/14 (from 10),
  layout 39 10/14 (the best free setup there is also 10/14). Median position error 2.0 cm (from 2.8)
  and 1.9 cm (from 2.2). Details in "Close looks" below.
- **What limits each detector:**
  - **exact:** only what the route cannot reach (drawers too low for the arm, one cabinet it cannot
    open, one microwave out of reach)
  - **camera detectors:** mostly naming, and objects that cover too few pixels (under about 300
    px they are rarely found; over 3000 px both keep about 85%)
- **Positions:** a typical error is a few cm, from turning a box into a point with depth. It does
  not shrink much with distance.
- **The VLM runs on layouts 7, 25 and 39** used the earlier route: no cabinets or drawers opened,
  and the arm blocked the fridge. No VLM run was made on the current route, to save cost.

## Close looks

The question: in the failure pictures the object is far, off to the side, or partly hidden. Would
looking from close by, and moving what is in the way, fix it?

**What the failures show** (replay of all 9 camera runs: 124 true objects, 65 found right, 59 not):

- No object, found or not, was ever seen from closer than 0.93 m. The base cameras cannot get
  closer: the base, plus the counter's depth. Distance alone does not tell hits from misses (median
  1.41 m against 1.46 m); the best viewing angle does more (8 degrees off the camera's axis against 14).
- 30 failures were seen unhidden but far or off-axis, mostly wrong names. 12 were partly hidden in
  every picture (against 7 of the 65 successes). 17 were never in a picture: the fork in a drawer,
  the plate in a cabinet, and, in the earlier-route VLM runs, things behind doors not opened.

**What was built** (free, no VLM; the robot uses only its own map, depth and detections):

- `look_closely` (chefbot/explore/look.py): for an object the detector is unsure of (seen only small,
  low score, two names on one spot, or only ever partly hidden by a nearer surface in the robot's own
  depth image), the robot stands about 0.65 m in front of it and moves the arm so the wrist camera is
  about 0.32 m from it, looking down 50 degrees with the object above the gripper in the picture.
  Straight on first, then 30 degrees to the left and right if no arm pose fits or it is still hidden.
  A close view's position wins, and its name counts 3 times in the name vote. In close views SigLIP 2
  names every box (OWLv2 names a big close-up badly: it called the lemon a mango), the gripper's own
  pixels are masked out from the robot's geometry, and image-sized boxes are dropped.
- Moving what is in the way (chefbot/explore/move_aside.py, off by default): if the object is still
  partly hidden from every side, the robot picks the mapped object in front of it, sets it down on a
  free spot of the same counter, looks again and puts it back. The grasp is the ground-truth
  proposer's geometry run on the robot's own depth points (completed by mirroring the seen half),
  not on the object's true mesh; the free spot comes from the depth images too. On a bench test it
  picked up the bottled water, the apple and the lemon; it found no grasp on the banana or the cup.

**Results** (OWLv2 best free setup, current route; run times with three runs sharing the Mac):

| | found | wrong name | median / mean xy error | close looks (failed: no arm pose) | objects moved | run time |
|---|---|---|---|---|---|---|
| layout 7, best free setup | 10/14 | milk, spoon | 2.8 / 4.7 cm | | | 15 min (alone) |
| layout 7, + close looks | **11/14** | milk | **2.0 / 4.4 cm** | 36 (10) | 0 | 36 min |
| layout 39, best free setup | 10/14 | bowl, milk | 2.2 / 3.1 cm | | | 27 min |
| layout 39, + close looks | 10/14 | bowl, milk | **1.9 / 2.5 cm** | 25 (1) | 0 | 40 min |

- The spoon on layout 7 is now named right; positions get a little better on both layouts.
- Nothing was moved: 8 and 3 close views still showed the object partly hidden, but in no case did a
  mapped object stand between the robot and them (what hid them was not something it had mapped).
- What close looks did not fix: the fork and plate (inside a drawer and a cabinet: never in a
  picture); the milk in the fridge, called bottled water (seen big and always by that name, so never
  uncertain and never looked at closely); and on layout 39 the bowl in the microwave, called a
  cutting board even from close by.
- Cost: about one extra stop per uncertain object, many of them on false finds (spatulas, ladles);
  the run takes about half again as long.

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
