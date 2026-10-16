
HANDHELD TOOLS AND POURING
- Some kitchen tools are used while held: pick the tool up, align to the container, then
  lower_into(container) and squeeze(). An egg cracker with an egg in it cracks the egg into
  whatever it is lowered over (put the egg in the cracker first: place(egg_cracker) while
  holding the egg). An electric whisk runs only while you squeeze and stops on release().
- Cracking leaves the eggshell in the cracker. Empty it before the next egg: pick up the
  cracker and pour() the shell into the trash can (or a spare bowl for scraps), never into
  the food.
- release() lets go of the trigger, not the tool; place puts the tool down.
- Release a running whisk before lifting it out (lift_out(), or driving off, which lifts it
  out): a running whisk splatters everything when it comes out of the bowl.
- A cooking spoon or spatula: pick it up, lower_into the bowl or pan, then stir(seconds).
  Stirring in a bowl mixes; stirring eggs in a pan while they cook scrambles them (eggs left
  unstirred set in one piece). Stir in turns of 20-30 s and check how it is going.
- pour(container) tips what the held bowl or pan contains into another container or onto
  a plate. Align to the target first.
