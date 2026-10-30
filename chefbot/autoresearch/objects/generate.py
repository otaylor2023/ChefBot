"""Run a generator (contract.md) for one seed, sandboxed: no network, read only its own directory and its Python,
write only the output directory, a time limit. The output directory gets model.xml (and its files), params.json and
gen_log.txt.

The generator's interpreter is GEN_PYTHON (numpy, scipy, trimesh, manifold3d, coacd, shapely, build123d, mujoco,
pillow, pyyaml); default this one.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from chefbot.autoresearch.episode import _sandbox_prefix, protected_paths

RUNNER = '''
import importlib.util, json, sys, traceback
gen_dir, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, gen_dir)
try:
    spec = importlib.util.spec_from_file_location("generate", gen_dir + "/generate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    params = mod.generate(seed, out)
    if not isinstance(params, dict):
        raise TypeError(f"generate() returned {type(params).__name__}, not a dict")
    with open(out + "/params.json", "w") as f:
        json.dump(params, f, indent=1)
except BaseException:
    traceback.print_exc()
    sys.exit(3)
'''


def gen_python() -> str:
    return os.environ.get("GEN_PYTHON", sys.executable)


def dir_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def run(gen_dir: Path, seed: int, out: Path, timeout_s: float, max_mb: float, extra_deny: list[str] = ()) -> dict:
    """Generate one asset into `out` (emptied first). Returns {ok, params, gen_s, error}."""
    gen_dir, out = Path(gen_dir).resolve(), Path(out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    os.chmod(out, 0o777)
    if not (gen_dir / "generate.py").exists():
        return {"ok": False, "error": "the generator has no generate.py"}
    runner_dir = Path(tempfile.mkdtemp(prefix="chefbot_gen_"))
    runner = runner_dir / "runner.py"
    runner.write_text(RUNNER)
    os.chmod(runner_dir, 0o755)
    os.chmod(runner, 0o644)
    python = gen_python()
    deny = protected_paths() + list(extra_deny)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(out), "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4", "MPLBACKEND": "Agg"}
    cmd = _sandbox_prefix(gen_dir, out, deny, runner, python) + [python, "-I", str(runner), str(gen_dir), str(seed), str(out)]
    t0 = time.time()
    log = open(out / "gen_log.txt", "w")
    try:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(out), timeout=timeout_s)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    finally:
        log.close()
        shutil.rmtree(runner_dir, ignore_errors=True)
    gen_s = round(time.time() - t0, 2)
    tail = (out / "gen_log.txt").read_text(errors="replace")[-1500:]
    if code == "timeout":
        return {"ok": False, "gen_s": gen_s, "error": f"generate() took more than {timeout_s:g} s"}
    if code != 0 or not (out / "params.json").exists():
        return {"ok": False, "gen_s": gen_s, "error": f"generate() failed (exit {code}): {tail.strip()[-600:]}"}
    mb = dir_mb(out)
    if mb > max_mb:
        return {"ok": False, "gen_s": gen_s, "error": f"the asset is {mb:.1f} MB (limit {max_mb:g} MB)"}
    try:
        params = json.loads((out / "params.json").read_text())
    except json.JSONDecodeError as e:
        return {"ok": False, "gen_s": gen_s, "error": f"params are not JSON: {e}"}
    return {"ok": True, "gen_s": gen_s, "params": params, "asset_mb": round(mb, 2)}
