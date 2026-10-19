
EXPLORING THE KITCHEN (this run)
- This run comes before any cooking. The map you were given has the kitchen's fixtures
  (MAP PLACES) but none of its loose objects: food, cookware, utensils, dishes, bottles.
  Your job is to find all of them, so later tasks start from a complete map. Do not pick,
  place or move anything.
- Look at every fixture in MAP PLACES (burners are part of the stove: the stove covers
  them). For each one: go_to it, then detect() with no classes: with no classes it looks
  for every kind of thing it knows. Everything detect finds is written into KITCHEN for
  you; you do not record it yourself.
- Fixtures with a door (fridge, microwave, dishwasher, oven, cabinets, drawers) hide what
  is inside. For each: detect its handle (e.g. detect(["fridge handle"])), align to the
  handle, open it, detect() again to see inside, then close it before moving on.
- A long counter may not be fully in view from one spot: complete names the part you
  have not seen; go_to an object you found near that part and detect() again from there.
- complete(step) tells you which fixtures you have not looked at or into yet, and which
  doors you left open. Keep going until it passes, then finish(summary) with a short list
  of what you found where.
