#!/usr/bin/env python
"""Pick a recipe from what's in the kitchen (the proactive step before extraction). No LLM, no network.

  python choose_recipe.py --corpus ../../recipe_corpus --pantry "eggs, tomatoes, green onions, sugar, rice" \
      [--diet vegetarian] [--exclude cilantro] [--max-missing 2] [--robot rank|strict|ignore] [--skills chefbot_sim] [-k 5]

Prints the top candidates with what they use, what is missing (to ask about, never to buy) and what the robot can't
do. Hand the chosen recipe to run_recipe.py for extraction and the exact per-step feasibility report.
"""
import argparse
import json
import sys
from pathlib import Path
from recipe_ir.actions import load_skills
from recipe_ir.choose import Prefs, choose, load_corpus

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--corpus", required=True)
ap.add_argument("--pantry", required=True, help="comma-separated items")
ap.add_argument("--diet", choices=["vegetarian", "vegan"])
ap.add_argument("--exclude", default="", help="comma-separated ingredients to avoid")
ap.add_argument("--max-missing", type=int, default=2)
ap.add_argument("--robot", default="rank", choices=["rank", "strict", "ignore"])
ap.add_argument("--skills", default="chefbot_sim")
ap.add_argument("-k", type=int, default=5)
ap.add_argument("--json", action="store_true")
a = ap.parse_args()
split = lambda s: [x.strip() for x in s.split(",") if x.strip()]
prefs = Prefs(diet=a.diet, exclude=split(a.exclude), max_missing=a.max_missing, robot=a.robot)
top = choose(load_corpus(Path(a.corpus)), split(a.pantry), prefs, load_skills(a.skills), k=a.k)
if a.json:
    print(json.dumps(top, indent=1))
    sys.exit(0)
if not top:
    print("No recipe fits these constraints. Loosen --max-missing or --robot, or add to the pantry.")
for i, t in enumerate(top, 1):
    print(f"{i}. {t['name']}   (robot fit {t['robot_fit']:.0%})")
    print(f"   uses: {', '.join(t['pantry_used'])}")
    print(f"   missing (ask, don't buy): {'; '.join(t['missing']) or 'nothing'}")
    if t["robot_blockers"]:
        print(f"   robot can't: {'; '.join(t['robot_blockers'])}")
