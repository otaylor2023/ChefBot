"""Shared configuration for standalone recipe extraction (not ChefBot execution)."""
from __future__ import annotations

import os

DEFAULT_RECIPE_MODEL = os.getenv("RECIPE_MODEL", "gpt-6.1-sol")
DEFAULT_REASONING_EFFORT = os.getenv("RECIPE_REASONING_EFFORT", "low")
