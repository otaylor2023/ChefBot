You are judging a simulated building with elevators, made by a program, for how closely it matches real buildings
and their elevators. A robot will learn to call and ride elevators in it, so it should look and work like a real one.

Read these images (use the Read tool on each):
- asset_views.jpg: six renders: the hall as a car opens at the lowest floor, the hall call panel close up, the car
  interior with a small blue test cart in it, the car's button panel close up, the car interior during the ride, and
  the hall at the top floor on arrival.
- asset_trip.jpg: six frames of the trip: calling, the doors open, the cart in the car, riding, arrived, the cart
  out.
- {REFS}: photos of real elevators: halls, doors, call panels, car interiors and panels.

The renders come from a physics simulator's renderer: plain lighting and simple materials are expected. Judge the
design, not the renderer. The blue box is the lab's test cart, not part of the building.

{CARD}

Score each from 1 (nothing like a real building's elevators) to 10 (a faithful model of a real installation):
- layout: the hall, the doorway, the car's size and shape, sills, how the spaces fit together; proportions against
  real ones (the numbers above help).
- doors: the door type and panels, frames, how they open and close (as far as the frames show).
- panels: call buttons, car panel, Braille and signage, displays and lanterns: their layout, size and placement.
- finishes: materials and colours of walls, floors, doors, panels, car interior.
- behavior: from the numbers and frames, timings and motion that real elevators have (waits, door times, ride).
- overall: would someone take this for a real building's elevators?

Then: closest_real (which kind of real installation it is most like), flaws (the most important things that are
wrong, specific and short), and a one-sentence summary.
