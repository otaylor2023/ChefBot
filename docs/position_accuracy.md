# Position accuracy of the explored semantic map

How close the explorer puts each object to where it really is, for the three detectors: the
exact simulator detector (`groundtruth`), the vision language model (`vision`) and OWLv2 (`owl`).
The scorer only asks whether an object is within 15 cm. This report measures the error itself.

## How it is measured

`chefbot/explore/accuracy.py` reads a finished run folder. Only this analysis reads the true map,
after the run. The agent and the detectors never see it.

- **Matching:** each true object is paired with the explored object of the same label nearest to
  it, within 0.5 m, each used once. Anything farther counts as not found.
- **Per object:**
  - xy, z and 3D error against the true map's position (the object's body origin)
  - the same errors against the middle of the object's geometry, which is what a detector sees
  - the reference offset between the two
  - whether the object was put in or on the right fixture
- **Per run:** matched count, mean, median, p90 and max xy error, the share within 5, 10 and
  15 cm, the 3D and z medians, and the share with the right fixture.
- **Heights:** runs now write `explored_positions.yaml` and `true_positions.yaml`. For the older
  runs:
  - The explored heights come from the detector's frame logs (`vision/`).
  - The true positions come from the same scene rebuilt and reset (`--rebuild-truth`).
    Placement is deterministic. An object whose rebuilt spot differs from the run's true map is
    left without a height. That only happens for the plate and fork of the hard task, whose
    placement changed when cabinets and drawers became explorable.

Run it:

```
python -m chefbot.explore.accuracy runs/<run> [runs/<run> ...] [--rebuild-truth] [--markdown]
```

## The reference offset

The true map stores each object's body origin. For every object but one, that is within 0.1 cm
of the middle of its geometry (median 0.06 cm). The exception is the **pan**: its origin sits
**10.7 cm** from its middle, toward the handle (0.1 cm in height). So about 10 cm of any pan error
is the true map's choice of reference point, not the detector. That is why the exact detector's
xy max is 10.7 cm on every kitchen, and its error against the geometry's middle is 0.0 cm.

## Results per run

Errors in cm. Kitchens: layout 1 is the default 6-object task, and layouts 7, 25 and 39 are the
hard 14-object task.

| run | detector | layout | matched | xy median | xy p90 | xy max | within 5/10/15 cm | 3D median | z median | xy vs centre median | fixture right |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-25_145153 | groundtruth | 1 | 6/6 | 0.1 | 5.4 | 10.7 | 83%/83%/100% | 0.1 | 0.0 | 0.0 | 100% |
| 2026-10-25_145308 | groundtruth | 7 | 13/14 | 0.1 | 0.1 | 10.7 | 92%/92%/100% | 0.1 | 0.0 | 0.0 | 100% |
| 2026-10-25_150237 | groundtruth | 25 | 12/14 | 0.0 | 0.3 | 10.7 | 92%/92%/100% | 0.1 | 0.0 | 0.0 | 100% |
| 2026-10-25_151243 | groundtruth | 39 | 12/14 | 0.0 | 0.1 | 10.7 | 92%/92%/100% | 0.1 | 0.0 | 0.0 | 100% |
| 2026-10-19_184125 | vision | 1 | 6/6 | 4.2 | 12.2 | 12.9 | 67%/67%/100% | 4.7 | 1.3 | 3.0 | 100% |
| 2026-10-20_144100 | vision | 7 | 10/14 | 4.0 | 10.8 | 12.5 | 60%/80%/100% | 4.0 | 1.6 | 3.2 | 100% |
| 2026-10-20_144336 | vision | 25 | 7/14 | 4.5 | 13.7 | 14.7 | 57%/71%/100% | 5.1 | 1.0 | 3.1 | 100% |
| 2026-10-20_144626 | vision | 39 | 8/14 | 2.5 | 10.1 | 20.1 | 62%/88%/88% | 2.9 | 1.2 | 2.5 | 100% |
| 2026-10-25_145221 | owl | 1 | 4/6 | 6.0 | 8.6 | 9.6 | 25%/100%/100% | 6.2 | 1.7 | 3.9 | 75% |
| 2026-10-20_104208 | owl | 7 | 8/14 | 6.2 | 13.5 | 20.5 | 38%/75%/88% | 6.0 | 1.2 | 4.8 | 100% |
| 2026-10-20_104434 | owl | 25 | 6/14 | 4.1 | 10.2 | 14.3 | 67%/83%/100% | 4.7 | 1.2 | 3.8 | 83% |
| 2026-10-20_104644 | owl | 39 | 6/14 | 5.6 | 15.9 | 20.1 | 50%/67%/83% | 5.7 | 1.4 | 5.6 | 100% |

## Results per detector (all four kitchens together)

| detector | matched | xy mean | xy median | xy p90 | xy max | within 5/10/15 cm | 3D median | 3D p90 | z median | xy vs centre median | fixture right |
|---|---|---|---|---|---|---|---|---|---|---|---|
| groundtruth | 43/48 | 1.1 | 0.1 | 0.3 | 10.7 | 91%/91%/100% | 0.1 | 1.1 | 0.0 | 0.0 | 100% |
| vision | 31/48 | 5.6 | 3.8 | 12.9 | 20.1 | 61%/77%/97% | 4.1 | 12.9 | 1.2 | 2.6 | 100% |
| owl | 24/48 | 6.8 | 5.9 | 13.5 | 20.5 | 46%/79%/92% | 6.0 | 14.0 | 1.3 | 4.5 | 92% |

## Current route: exact detector vs OWLv2 (layouts 7, 25, 39)

The current route opens cabinets and drawers, and each view uses the camera the arm blocks least
(the fridge is in view). Both detectors ran the same route, which makes this the fair comparison
for OWLv2. No VLM runs were made on this route.

| layout | detector | found | xy median | xy p90 | xy max | within 15 cm | 3D median | right fixture | extra finds | missed |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | exact | 13/14 | 0.1 | 0.1 | 10.7 | 100% | 0.1 | 100% | 0 | fork |
| 7 | owl | 9/14 | 3.0 | 9.8 | 10.8 | 100% | 3.6 | 89% | 22 | egg, fork, lemon, milk, plate |
| 25 | exact | 12/14 | 0.0 | 0.3 | 10.7 | 100% | 0.1 | 100% | 0 | bowl, fork |
| 25 | owl | 6/14 | 5.4 | 14.1 | 20.8 | 83% | 5.8 | 100% | 14 | bowl, egg, fork, lemon, milk, mug, plate, spoon |
| 39 | exact | 12/14 | 0.0 | 0.1 | 10.7 | 100% | 0.1 | 100% | 0 | fork, plate |
| 39 | owl | 6/14 | 4.3 | 15.3 | 21.6 | 83% | 4.2 | 100% | 26 | bowl, cup, egg, fork, lemon, milk, plate, spoon |

All three layouts together:

| detector | found | xy mean | xy median | xy p90 | xy max | within 5/10/15 cm | 3D median | xy vs centre median | right fixture | extra finds |
|---|---|---|---|---|---|---|---|---|---|---|
| exact | 37/42 | 0.9 | 0.1 | 0.2 | 10.7 | 92%/92%/100% | 0.1 | 0.0 | 100% | 0 |
| owl | 21/42 | 6.0 | 3.6 | 10.8 | 21.6 | 57%/86%/90% | 4.0 | 3.0 | 95% | 62 |

- Opening the cabinets and drawers and clearing the camera did not raise OWLv2's share much:
  21/42 here, against 20/42 on the earlier route.
- What OWLv2 misses that the exact detector saw: the egg and milk in the fridge (all three
  kitchens, even with the shelves in view), the lemon (three times), the plate in its cabinet
  (layouts 7 and 25), and the bowl, mug, spoon and cup once or twice. These are small objects, or
  objects that score below the detector's threshold (the egg scores 0.15 against a threshold of
  0.2).
- Its extra finds (62 over three kitchens) are utensils, decorations and fixture parts named as
  objects, many of them seen inside the open cabinets.
- Placement of what it finds is as good as before: median 3.6 cm, 90% within 15 cm.

Runs: exact `2026-10-25_145308`, `150237`, `151243`; OWLv2 `2026-10-25_175806`, `181121`,
`182428` (fixtures `owl_current_layout007/025/039`).

## What it says

- **The exact detector sets the floor.** Its only error is the pan's reference offset. What it
  misses (5 of 48) is what the route never looks at: drawers the arm cannot reach, a cabinet it
  cannot open.
- **The VLM places what it finds well.** Median error is about 4 cm (2.6 cm against the middle of
  the geometry), 97% of its finds are within 15 cm, and every one is on the right fixture.
  Heights are good (median 1.2 cm). The xy error comes mostly from depth back-projection of a box
  around the object: the box includes background, and parts of the object are hidden.
- **OWLv2 finds fewer objects and places them a little worse:** median about 6 cm, 92% within
  15 cm, and two objects on the wrong fixture.
- **Caveat:** the vision and owl runs on layouts 7, 25 and 39 are from the earlier route. It did
  not open cabinets or drawers, and the front camera had the arm blocking the fridge. So their
  matched counts are lower than the current route would give, but the position errors of what
  they found are comparable. Layout 1 was the same route for all three detectors.

## Tests

`chefbot/tests/explore/test_accuracy.py`:

- **Unit tests** for matching (label, nearest first, the 0.5 m radius), the per-object errors
  (origin, centre, offset, fixture) and the summary statistics, on small hand-made maps.
- **Regression tests** on copies of the fifteen runs above (`chefbot/tests/explore/fixtures/accuracy/`).
  They assert per-detector bounds set from these numbers with some margin: matched share, xy
  median, xy p90, 3D median, and the share within 15 cm. A further test checks that the exact
  detector's error against the geometry's middle stays within 1 cm.
- **An opt-in slow test** reruns OWLv2 on layout 1 (free, about 2 minutes) and checks the same
  bounds: `CHEFBOT_SLOW=1 python -m pytest chefbot/tests/explore/test_accuracy.py -m slow`.
