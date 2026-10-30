# The tongs generator contract

A **generator** is a directory (`generators/<name>/`) with a `generate.py` and a `ranges.yaml`. The lab runs it
once per seed, each run in a fresh output directory, in a sandbox (no network, no files outside the generator's
own directory and the output directory, a time limit per seed). Each seed must give **one pair of kitchen tongs**,
and different seeds should give different tongs, the way different real products differ.

## generate.py

```python
def generate(seed: int, out_dir: str) -> dict:
    """Write out_dir/model.xml (and any meshes and textures it uses, under out_dir). Return the tongs' parameters."""
```

- Deterministic: the same seed gives the same tongs.
- At most **{GEN_TIMEOUT_S} s** and **{MAX_ASSET_MB} MB** of files per seed.
- Its Python: {GEN_PACKAGES}. Anything else must be in the generator's own directory (pure Python or data
  files you made). Downloaded meshes or images can't be the tongs themselves: the generator has to make the
  shape from its parameters.
- `print` goes to the build log (`gen_log.txt`).

## model.xml (MuJoCo {MUJOCO_VERSION})

One `<mujoco>` file. The lab composes it into its own scenes, so:

- **`worldbody` holds exactly one body, named `object`**, at the origin, with **no joint** (the lab adds a free
  joint). Its origin is the **centre of the tongs' bounding box at rest**: a kitchen places an object by that point.
- Inside `object`: two bodies, **`arm_a` and `arm_b`**. Each holds one arm and everything rigidly fixed to it
  (its tip, its half of the handle). Anything fixed to neither arm (a rivet, a lock tab's pivot) may stay in `object`.
- **Each arm moves relative to `object` on its own hinge.** MuJoCo ignores contacts between a body and its parent,
  so an arm fixed in `object` would never touch an arm hinged to it: the tips would pass through each other. Make
  both arms children of `object`, each with a hinge (couple them with an `<equality><joint>` if you want them to
  move as a mirror pair).
- **The spring** that holds the tongs open: joint `stiffness` with `springref`, and joint limits (`range`) for the
  open stop. Limits in MuJoCo are soft; a preloaded spring sinks past a soft stop, so make the limit stiff
  (`solreflimit`) or account for it.
- **Sites** the lab uses (names exact):
  - `grip_a`, `grip_b`: where a hand squeezes the handles, on each arm's outer surface. The lab pushes there.
  - `tip_a`, `tip_b`: the centre of each tip's gripping surface.
- **Collision geoms** (`group="0"` or `"3"`) must follow the real shape closely: a mesh collides as its convex
  hull, so a concave part (a scalloped tip, a curved channel) must be split into convex pieces (CoACD is
  installed) or built from primitives. At most {MAX_COLLISION_GEOMS} collision geoms and {MAX_COLLISION_VERTS}
  collision-mesh vertices in all.
- **Visual geoms** (`group="1"` or `"2"`, `contype="0" conaffinity="0"`): what cameras see. They may be finer than
  the collision geoms. The lock and its pull tab are visual only (no collision, no joint).
- **Mass:** give every geom an explicit `mass`: real-material masses on the collision geoms, `mass="0"` on visual
  geoms (MuJoCo gives every geom mass by default, visual ones too; RoboCasa overrides density, not explicit mass).
- **Contact:** per-geom `friction`, and `solref`/`solimp` where a material is soft (silicone, nylon). These are
  the generator's to set; the lab keeps them.
- **Angles are radians** (as in RoboCasa): put `<compiler angle="radian"/>` in model.xml. Mesh and texture paths
  are relative to `out_dir`. No `<include>`, no plugins, no `<keyframe>`. Nothing outside
  `worldbody` other than `asset`, `default`, `equality`, `contact` (`exclude`/`pair`) and `tendon`.
- The arms must not overlap at rest.

## The returned parameters

A flat dict of JSON values, in physical units. Required:

| key | meaning |
|---|---|
| `length_m` | overall length, tips to hinge end |
| `mass_kg` | total mass |
| `rest_tip_gap_m` | distance between `tip_a` and `tip_b` at rest (open) |
| `closing_force_N` | force on each of `grip_a`/`grip_b`, pushing them together, that just brings the tips into contact |
| `tip_friction` | the tips' sliding friction coefficient |
| `style` | a few words: tip type, materials, lock (e.g. "scalloped silicone tips, stainless arms, pull-ring lock") |

Add any others you generate from (arm width, tip shape, spring stiffness, colours, ...). The lab measures
`length_m`, `mass_kg`, `rest_tip_gap_m` and `closing_force_N` in simulation and compares them with what you declare.

## ranges.yaml

Every parameter the generator draws, its range and distribution, and **why**: what real products or sources the
range comes from.

```yaml
length_m: {min: 0.18, max: 0.42, dist: "uniform over {0.23, 0.30, 0.35, 0.40} +- 0.01", why: "common lengths: 9, 12, 14, 16 in (several manufacturers' catalogs)"}
closing_force_N: {min: 3, max: 20, dist: ..., why: ...}
```

Each generated asset's parameters must fall inside these ranges (the lab checks).
