# Why the camera detectors miss or misplace objects

A root cause analysis of the VLM and OWLv2 detectors, over the saved exploration runs. No new VLM
calls were made.

## Method

The run logs keep every frame a detector saw (`vision/NNN.png`), what it named in it and where
that put things (`vision/NNN.json`). They do not keep the camera pose, the depth image or what was
hidden. The scripted route is deterministic, so `chefbot/explore/rootcause.py` replays each run's
tool calls in the same scene and stands the robot exactly where the run logged it. At every call
that produced a detector frame, it renders the same camera with RGB-D and a segmentation image.

Each logged frame is matched to the replayed view by image. The replayed images are identical to
the logged ones (mean pixel difference 0.07 to 2.7 out of 255). 361 of 363 frames matched; the 2
unmatched ones are left out.

For every true object in every frame, the sighting table (`docs/root_cause/sightings.csv`,
1085 rows) records:

- whether it was in the camera's view, and how many pixels of it were visible (hidden ones count 0)
- its distance from the camera, the viewing angle, its physical size, and whether it touched the
  image edge
- the detector's outcome:
  - **kept:** a detection of its label within 0.3 m of it, with the position error
  - **dropped:** named right, but then thrown away by the edge filter, by depth landing on the
    background, or by the other filters
  - **wrong label:** a box over it, named something else
  - **not named:** visible, but no box over it
- OWLv2's score for the right label over the boxes on it, below the threshold too (OWLv2 rerun
  locally on every saved frame)

Per object, `docs/root_cause/objects.csv` gives the furthest stage it reached over all its frames,
and so why it was missed.

Runs:

| detector | runs | route |
|---|---|---|
| VLM | `2026-10-19_184125` (layout 1), `2026-10-20_144100`, `144336`, `144626` (layouts 7, 25, 39) | earlier route: no cabinets or drawers opened, front camera (the arm blocks the fridge) |
| OWLv2 | `2026-10-25_145221` (layout 1), `2026-10-25_175806`, `181121`, `182428` (layouts 7, 25, 39) | current route: cabinets and drawers opened, the camera the arm blocks least |

## 1. Where objects get lost

Each object of each run (48 per detector):

| cause | VLM | OWLv2 | objects |
|---|---|---|---|
| found | 31 | 25 | |
| never visible | 10 | 5 | VLM: egg, milk (fridge blocked by the arm), plate, fork, bowl. OWLv2: plate, fork (cabinet or drawer it cannot open or reach), bowl (microwave out of reach) |
| visible, but named something else | 4 | 9 | VLM: milk as bottled water or bottled drink, bowl as plate, cup as mug. OWLv2: egg as bar soap (4), milk as bottled water (3), lemon as pickle or apple, bowl as cutting board |
| kept in some frames, then lost in the map | 0 | 5 | OWLv2: spoon (became fork, or pizza cutter and scissors), cup and mug (swapped), lemon (became sponge) |
| visible, too few pixels (under 300) | 2 | 2 | VLM: egg (136 px), pan (83 px). OWLv2: egg, plate |
| named right, then dropped by the edge filter | 1 | 1 | apple, mug |
| visible, not named | 0 | 1 | plate |

- **VLM:** most misses were never visible (10 of 17). That is the earlier route; the current route
  opens cabinets and drawers and uses a camera with the fridge in view. Its visible misses are
  mostly labels (4) and too few pixels (2).
- **OWLv2:** most misses are visible but named wrong (9 of 23), or kept and then lost in the map
  (5).

## 2. Pixels and distance

Per sighting (an object visible in one frame):

| visible pixels | VLM sightings | VLM kept | VLM wrong label | OWLv2 sightings | OWLv2 kept | OWLv2 wrong label |
|---|---|---|---|---|---|---|
| 0 to 100 | 6 | 17% | 17% | 165 | 7% | 31% |
| 100 to 300 | 22 | 55% | 18% | 191 | 30% | 46% |
| 300 to 1000 | 61 | 62% | 21% | 173 | 42% | 43% |
| 1000 to 3000 | 23 | 61% | 4% | 76 | 58% | 11% |
| 3000 and more | 15 | 87% | 0% | 26 | 85% | 4% |

![miss rate vs pixels](root_cause/miss_rate_vs_pixels.png)

| camera to object | VLM kept | OWLv2 kept | median visible px (OWLv2) |
|---|---|---|---|
| 1.0 to 1.5 m | 72% | 53% | 655 |
| 1.5 to 2.0 m | 45% | 39% | 377 |
| 2.0 to 3.0 m | 58% | 31% | 146 |
| over 3 m | 50% (4 sightings) | 7% | 65 |

- **Miss rate falls steadily with pixel size, with a clear knee.** Under 300 px, objects are
  rarely kept (VLM 45%, OWLv2 under 30%). Above 3000 px, both keep about 85 to 87%.
- **OWLv2's wrong labels are a pixel problem.** It names 43 to 46% of sightings under 1000 px as
  something else, against 11% at 1000 to 3000 px. The VLM is less sensitive: about 20% wrong
  labels up to 1000 px, then near 0.
- **Mean distance:**
  - VLM: kept 1.56 m (734 px median), not kept 1.75 m (417 px).
  - OWLv2: kept 1.76 m (447 px), not kept 2.83 m (147 px).
- Distance matters through pixels. At the robot's usual standoff (1.2 to 1.5 m), a lemon, egg or
  spoon covers only about 100 to 500 px.

![pixels vs distance](root_cause/pixels_vs_distance.png)

## 3. Position error

The median error of kept sightings is 4.2 cm (VLM) and 3.9 cm (OWLv2). It barely depends on
distance (correlation +0.27 VLM, +0.16 OWLv2) and not at all on pixel size (+0.05, +0.03).

The error comes from turning a box into a point: the box centre's depth includes background and
the visible side of the object, not its middle. Getting closer would not reduce it much; a
tighter object mask would.

Of the sightings named right but dropped later:

- **edge filter:** VLM 18, OWLv2 25
- **depth at the box centre landed on the background:** VLM 5, OWLv2 4
- **depth hit the object, but a later filter dropped it:** VLM 1, OWLv2 28. These are the
  height, footprint and 4 m range filters; most of OWLv2's are views from 4 to 6 m away, beyond
  the range limit.

![error vs distance](root_cause/error_vs_distance.png)

## 4. Model quality

- **OWLv2 scores for the right label, on objects in view:**
  - median 0.265
  - 61% at or above its 0.2 threshold, 69% at 0.15, 79% at 0.10
  - per label: lowest for the plate (0.03), milk (0.11), bottled water (0.12) and lemon (0.18);
    highest for the cutting board (0.51) and mug (0.43)
- **Its confusions are systematic, not noise:**
  - bottled water as condiment bottle (23 sightings)
  - cup as mug (21)
  - banana as sponge (19)
  - lemon as pear (18) or avocado (16)
  - spoon as knife (12)
  - egg as bar soap

  Lowering the threshold would also admit more false finds (22 extra finds at 0.15 on layout 1).
- **The VLM's wrong labels are mostly near-synonyms:** cup and mug (6), milk as bottled water or
  bottled drink (2), apple as tomato (2), pan as bowl (2), lemon as lime or avocado.

## 5. What would recover objects, ranked

| fix | what it addresses | expected gain | cost |
|---|---|---|---|
| 1. Label vote per spot: keep the label seen most often at a spot, instead of "a closer view's name wins" | OWLv2 kept the right label in many frames (up to 32) and lost it to one other name | OWLv2 +5 of 48 objects (10%); VLM 0 here | small code change in the tracker, free |
| 2. Zoom on small or uncertain boxes: run the detector again on an enlarged crop around each box under about 1000 px, or with a low-margin label | too few pixels; OWLv2's wrong labels under 1000 px | OWLv2 up to about 6 to 9 more (eggs, lemons, milk, small plate), VLM about 2 to 4 (egg, pan, cup, milk), if crops behave like the 1000 to 3000 px band (58 to 61% kept, 4 to 11% wrong label) | one extra detector call per small box (VLM about $0.004 each; OWLv2 free) |
| 3. Go closer to small or uncertain objects (approach to about 0.6 to 0.8 m, or use the wrist camera) | the same objects as fix 2, with real pixels instead of upsampled ones | similar to fix 2, and more reliable for labels | a new action per candidate (time; navigation near counters) |
| 4. A better or bigger model, or a stricter prompt | VLM near-synonyms (cup and mug, milk); OWLv2 fine-grained confusions | VLM about +2 to 3 (milk, bowl, cup); OWLv2 depends on the model | a different model, prompt work, or a cup and mug equivalence in the scorer |
| 5. Re-look instead of the edge filter dropping a box cut off by the image edge | right labels dropped at the edge (VLM 18 sightings, OWLv2 25) | 1 object each here (other views usually catch it) | small: turn a little and detect again |
| 6. A mask (e.g. SAM) instead of the box centre for depth | position error (about 4 cm) and depth landing on the background | error down, and up to about 9 dropped sightings back | one segmentation model call per box |

- **Fixes 1 and 2 first.** They are cheap and address the largest visible losses. Fix 1 is free
  and applies to both detectors.
- **Getting closer (fix 3)** pays off mainly for small objects, and mainly for labels. It does not
  reduce position error, which is about the same at every distance.
- **A better model (fix 4)** matters most for OWLv2, whose confusions persist even at 300 to
  1000 px. The VLM's remaining label errors are near-synonyms, which a stricter prompt or a scorer
  equivalence (cup and mug) would largely cover.
- **On the current route,** "never visible" is down to drawers too low for the arm and a cabinet
  the robot cannot open. That is a reach problem, not a perception one.

## Reproduce

```
python -m chefbot.explore.rootcause runs/<run> [runs/<run> ...] --out <folder> --owl-scores
python -m chefbot.explore.rootcause runs/<run> [...] --out <folder> --analyze
```

The first writes `sightings.csv`. The second writes the tables, `objects.csv` and the three plots.
