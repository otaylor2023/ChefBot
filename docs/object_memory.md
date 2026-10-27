# Object memory

`chefbot/agent/world/objects.py` keeps one record per loose object, next to the semantic map.
It is built only from the agent's own tool calls, detections and notes, never from ground truth,
and it makes no model calls.

## 1. Identity: which one of like things

Five forks are five instances, `fork_1` .. `fork_5`, under the names the map gives them. A name
stays with its object:

* when you pick it and put it down (pick / place keep the name, as before);
* when it is seen again. Detections are matched to map entries by a cost: distance in metres plus
  0.3 m times how unlike the instance the sighting looks (`nav._LOOK_M`). How it looks is an
  appearance signature (a unit image embedding, the mean of its sightings) and attributes such as
  `{"color": "red"}`, when the detector gives them. Without them, place alone decides, as before.

The VisionDetector passes each box's SigLIP 2 image embedding on as its signature with
`vision.appearance_signature: true` (it needs `owl.hybrid_siglip`, which loads SigLIP 2; local and
free). `ObjectMemory.distinguishing(id)` lists the attributes no like one shares (how to tell it apart).

Limits: two truly identical objects (same model, same look) that someone swaps while the robot is
not looking cannot be told apart by any camera; the memory then follows place, and logs
`seen_moved` when an object turns up away from where you put it.

## 2. Usage and states

Every pick, place, lower_into, stir, squeeze and pour is logged on the object used and on the
object it was used on (`stirred` on the fork, `stirred_in` on the bowl), with the run time, the
active plan step and the dish (the plan goal). Effects (`EFFECTS`): a tool lowered, stirred or
squeezed becomes dirty; pouring empties the held container and fills the target, and moves its
contents; an egg squeezed in the cracker is used up and joins the contents of the bowl below.

States come from the object's category (`categories.yaml`): `clean` (clean / dirty), `fill`
(empty / some / full), `prep` (raw / cut / cooked / used up). Each records where it came from
(`default`, `you`, `seen`, `note`).

Plans: `note_object(id, for_step, ingredient)` sets an object aside for a later step (the dish is
filled in from the plan); `done=true` releases it.

## 3. Queries

Agent tools (MEMORY_SCHEMAS in the registry, no look needed):

* `find_objects(category, label, state, used, recipe, step, free)`: all filters optional; with
  none, the objects by category.
* `object_info(id)`: everything about one object, with its full history.
* `note_object(id, state, for_step, ingredient, note, done)`.

In code: `nav.objects.find(...)`, `nav.objects.usage("cutlery")` (which fork was used doing what),
`nav.objects.describe(id)`. The KITCHEN block shows a short tag after an object once something is
known (`fork_1 [dirty; for s5 (eggs)]`); defaults are not shown, so the prompt does not grow.

## Saving

`run_agent` writes `runs/<run>/object_memory.yaml`; `--objects <that file>` starts a later run
from it.
