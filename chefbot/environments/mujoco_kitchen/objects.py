"""Real masses for the kitchen's objects. RoboCasa gives every geom (visual ones too) a density of 100 kg/m^3: an egg
weighs 13 g, a pan 33 g. Here an object's collision geoms take its category's density, its visual geoms none, and the
result is clamped to the category's range of real masses (a convex decomposition overfills thin walls).

    MASS[category] = (density kg/m^3, min kg, max kg)
"""

from __future__ import annotations

MASS = {
    # produce and food (USDA FoodData Central typical weights; densities of whole produce, ~0.85-1.1)
    "egg": (1030, 0.045, 0.070),
    "tomato": (950, 0.06, 0.20),
    "peach": (950, 0.10, 0.22),
    "apricot": (950, 0.03, 0.06),
    "strawberry": (900, 0.010, 0.035),
    "kiwi": (1050, 0.06, 0.11),
    "banana": (950, 0.10, 0.20),
    "mango": (1000, 0.20, 0.45),
    "avocado": (1000, 0.15, 0.30),
    "potato": (1080, 0.10, 0.35),
    "apple": (850, 0.12, 0.25),
    "lemon": (900, 0.06, 0.15),
    "lime": (950, 0.04, 0.09),
    "orange": (900, 0.12, 0.25),
    "onion": (950, 0.10, 0.25),
    "donut": (300, 0.04, 0.09),
    "croissant": (200, 0.04, 0.08),
    # cookware (catalog weights: a 10-12 in skillet 0.9-1.6 kg; ceramic bowls and plates 0.3-0.7 kg)
    "pan": (2700, 0.8, 1.6),
    "bowl": (2300, 0.25, 0.7),
    "plate": (2300, 0.3, 0.7),
}
DEFAULT = (900, 0.02, 1.0)
CONTAINERS = ("pan", "bowl", "plate")
