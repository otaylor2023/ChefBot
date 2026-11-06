#!/usr/bin/env python
"""Compile a run's actions.json into a ChefBot job file (chefbot/jobs/*.yaml format).

  python to_job.py runs/<run>/actions.json --out scrambled_eggs_from_web.yaml [--name NAME] [--drop-unsupported]

The job's steps keep the recipe's words; done_when comes from each step's completion cue; the skills report goes
into the hints and source.needs_a_person. Add a kitchen-specific `symbolic:` block before running it in simulation.
"""
import argparse
import json
from pathlib import Path
import yaml
from recipe_ir.to_job import compile_job

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("actions")
ap.add_argument("--out")
ap.add_argument("--name")
ap.add_argument("--drop-unsupported", action="store_true")
a = ap.parse_args()
job = compile_job(json.loads(Path(a.actions).read_text()), a.name, include_unsupported=not a.drop_unsupported)
text = yaml.safe_dump(job, sort_keys=False, allow_unicode=True, width=120)
if a.out:
    Path(a.out).write_text(text)
print(text)
