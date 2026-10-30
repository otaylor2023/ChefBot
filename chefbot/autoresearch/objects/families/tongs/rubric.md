You are judging a simulated pair of kitchen tongs, made by a program, for how closely it matches real kitchen tongs.
It will be used to train a robot to use tongs, so it should look and work like a real product.

Read these images (use the Read tool on each):
- asset_views.jpg: six renders of the simulated tongs (on a counter from three-quarter, from the top over a grid of
  5 cm squares, from the side; the tips close up; the hinge end close up; squeezed shut by forces at the handles,
  red arrows).
- asset_squeeze.jpg: a strip of frames as the handles are squeezed (red arrows, force in newtons) and released.
- {REFS}: photos of real kitchen tongs, for comparison.

The renders come from a physics simulator's renderer: plain lighting and simple materials are expected. Judge the
object, not the renderer.

{CARD}

Score each from 1 (nothing like real tongs) to 10 (indistinguishable from a real product's design):
- shape: the arms' form (channel or strip, curves, taper), the handles, how they join.
- tips: the gripping ends (scallops, silicone or nylon heads, cups), their size and how they meet when closed.
- mechanism: the hinge and spring, the lock and pull tab; whether opening and closing look as real tongs do.
- materials: colours and finishes for what the parts would be made of.
- proportions: length, width, opening, and thickness against real tongs (use the 5 cm grid and the numbers above).
- overall: would someone take this for a real product's design?

Then: closest_real (which kind of real tongs it is most like), flaws (the most important things that are wrong,
specific and short), and a one-sentence summary.
