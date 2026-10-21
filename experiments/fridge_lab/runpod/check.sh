#!/usr/bin/env bash
# Pre-flight checks on the pod (as root, from the repo root): bash experiments/fridge_lab/runpod/check.sh
# Starts the vision server if it is not running (in tmux session "vision").
set -uo pipefail
REPO="$(pwd)"
LAB="${FRIDGE_LAB_ROOT:-/opt/fridge_lab}"
source "$LAB/env.sh"
SIM="$LAB/venvs/sim/bin/python"
ok() { echo "  OK   $*"; }
bad() { echo "  FAIL $*"; FAILED=1; }
FAILED=0

echo "== keys"
[ -f "$REPO/.env" ] && grep -q GEMINI_API_KEY "$REPO/.env" && ok ".env has GEMINI_API_KEY" || bad ".env missing or has no GEMINI_API_KEY"

echo "== isolation (as researcher / fridgeskill)"
for p in "$REPO/.env" "$REPO/experiments/fridge_lab/scenes.json" "$LAB/private" "$LAB/venvs/sim" /root; do
  [ -z "$p" ] && continue
  for u in researcher fridgeskill; do
    if setpriv --reuid=$u --regid=$u --init-groups -- test -r "$p" 2>/dev/null && setpriv --reuid=$u --regid=$u --init-groups -- ls "$p" >/dev/null 2>&1; then
      bad "$u can read $p"
    else ok "$u cannot read $p"; fi
  done
done 2>/dev/null
setpriv --reuid=researcher --regid=researcher --init-groups -- ls /opt/dev/robocasa >/dev/null 2>&1 && bad "researcher can list robocasa" || ok "researcher cannot list robocasa (test kitchens)"
setpriv --reuid=researcher --regid=researcher --init-groups -- cat "$REPO/experiments/fridge_lab/kitchen.py" >/dev/null 2>&1 && ok "researcher can read the lab source" || bad "researcher cannot read the lab source"

echo "== rendering ($MUJOCO_GL)"
"$SIM" - <<'EOF' && ok "kitchen builds and renders" || bad "kitchen build/render"
import time
from experiments.fridge_lab import splits
from experiments.fridge_lab.kitchen import LabHost
s = splits.load()["scenes"][0]
t = time.time(); h = LabHost(s["layout"], s["style"], "/tmp/fl_check"); print(f"    build {time.time()-t:.1f}s")
h.find_start(s["door"], 0)
t = time.time(); [h.rgbd() for _ in range(5)]; print(f"    rgbd x3 cameras: {(time.time()-t)/5*1000:.0f} ms")
EOF

echo "== vision server"
if ! curl -s -X POST localhost:8777/health -d '{}' >/dev/null; then
  tmux new -d -s vision "source $LAB/env.sh; $LAB/venvs/vision/bin/python $REPO/experiments/fridge_lab/vision_server.py --preload 2>&1 | tee $LAB/vision.log"
  for i in $(seq 1 120); do curl -s -X POST localhost:8777/health -d '{}' >/dev/null && break; sleep 2; done
fi
curl -s -X POST localhost:8777/health -d '{}' | grep -q cuda && ok "vision server on cuda" || bad "vision server not on cuda ($(curl -s -X POST localhost:8777/health -d '{}'))"

echo "== Gemini through the proxy"
"$SIM" - <<'EOF' && ok "gemini call metered" || bad "gemini call"
from pathlib import Path
import numpy as np
from experiments.astranomy.host import REPO
from experiments.ortho_grasp.query.models import load_env
load_env(REPO / ".env")
from experiments.fridge_lab import models as M
L = M.Ledger(Path("/tmp/fl_check_ledger.jsonl"), 0.05)
r = M.gemini("gemini-3.1-flash-lite", "Say OK.", [np.zeros((64, 64, 3), np.uint8)], 16, 0, None, None, L, 0.05, "check")
print("    ", r["text"].strip()[:20], f"${r['cost_usd']:.5f}", f"{r['latency_s']:.1f}s")
EOF

echo "== skill sandbox + one episode through the lab code"
"$SIM" - <<'EOF' && ok "episode ran" || bad "episode"
import json, shutil
from pathlib import Path
from experiments.fridge_lab import splits, models as M
from experiments.fridge_lab.episode import run_skill_episode, sandbox_kind
from experiments.fridge_lab.kitchen import LabHost
import os
skill = Path("/tmp/fl_check_skill"); shutil.rmtree(skill, ignore_errors=True)
shutil.copytree(Path("experiments/fridge_lab/example_skill"), skill); os.system(f"chmod -R a+rX {skill}")
(skill / "probe.py").write_text("")
s = splits.load()["scenes"][0]
h = LabHost(s["layout"], s["style"], "/tmp/fl_check_ep")
r = run_skill_episode(h, s, 0, skill, Path("/tmp/fl_check_ep"), M.Ledger(Path("/tmp/fl_check_ledger.jsonl"), 0.05), M.Vision(os.environ.get("FRIDGE_VISION_URL")))
print("    sandbox:", sandbox_kind(), "| end:", r["end"], "| robot time", r["robot_time_s"], "s | wall", r["wall_s"], "s")
print(open("/tmp/fl_check_ep/skill_log.txt").read()[:600])
assert r["end"] == "skill returned"
EOF

echo "== codex (as researcher)"
if setpriv --reuid=researcher --regid=researcher --init-groups -- env HOME=/home/researcher codex login status 2>&1 | grep -qi "logged in"; then
  ok "researcher is logged in to Codex"
else bad "run: su - researcher -c 'codex login --device-auth'"; fi
if setpriv --reuid=researcher --regid=researcher --init-groups -- env HOME=/home/researcher codex sandbox linux -- true >/dev/null 2>&1; then
  ok "Codex's own sandbox works here: use --codex-sandbox workspace-write"
else echo "  NOTE Codex's sandbox does not work in this container: use --codex-sandbox none (the researcher user still has no access to private files)"; fi

echo
[ "$FAILED" = 0 ] && echo "All checks passed." || echo "Some checks FAILED (above)."
