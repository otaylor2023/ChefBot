#!/usr/bin/env bash
# One-time setup of the fridge lab on a RunPod GPU pod (Ubuntu + CUDA, as root).
# Run from the ChefBot repo root (e.g. /opt/ChefBot):
#     bash experiments/fridge_lab/runpod/setup.sh
# Everything lives on the container disk (/opt): RunPod's /workspace volume ignores permissions. Needs ~80 GB (RoboCasa: ~23 GB).
# Afterwards: copy .env into the repo, and log Codex in as the researcher (see README.md here).
set -euo pipefail

REPO="$(pwd)"
LAB="${FRIDGE_LAB_ROOT:-/opt/fridge_lab}"
DEV="${FRIDGE_DEV:-/opt/dev}"
ROBOSUITE_COMMIT=5ce6643f3092639d08f7b0f90ed1c6a84f50552c
ROBOCASA_COMMIT=456174f62b89b8fca99eaaf33949c29fec9cfc2a
test -f "$REPO/pyproject.toml" && test -d "$REPO/experiments/fridge_lab" || { echo "run from the ChefBot repo root"; exit 1; }
mkdir -p "$LAB/venvs" "$DEV"

echo "== system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q git curl ca-certificates ffmpeg tmux util-linux bubblewrap \
    libegl1 libgl1 libglvnd0 libgles2 libosmesa6 libglfw3 libglib2.0-0 >/dev/null

echo "== node + codex"
if ! command -v codex >/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null
  apt-get install -y -q nodejs >/dev/null
  npm install -g @openai/codex >/dev/null
fi
codex --version

echo "== uv + python 3.11"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11

echo "== robosuite + robocasa (pinned)"
[ -d "$DEV/robosuite" ] || git clone -q https://github.com/ARISE-Initiative/robosuite.git "$DEV/robosuite"
[ -d "$DEV/robocasa" ] || git clone -q https://github.com/robocasa/robocasa.git "$DEV/robocasa"
git -C "$DEV/robosuite" checkout -q "$ROBOSUITE_COMMIT"
git -C "$DEV/robocasa" checkout -q "$ROBOCASA_COMMIT"

echo "== sim venv"
[ -d "$LAB/venvs/sim" ] || uv venv -q --python 3.11 "$LAB/venvs/sim"
SIM="$LAB/venvs/sim/bin/python"
uv pip install -q --python "$SIM" "mujoco==3.3.1" "mink==0.0.13" "numpy==2.2.5" "scipy==1.15.3" "numba==0.61.2" \
    "imageio[ffmpeg]==2.38.1" "lxml==6.1.3" "PyYAML==6.0.3" "pyyaml-include==1.4.1" "opencv-python-headless==5.0.0.93" \
    "pillow" "h5py" "gymnasium==0.29.1" "termcolor" "tqdm" "msgpack" "msgpack-numpy" "websockets" "qpsolvers" "quadprog" "daqp" \
    "coal" "ompl" "pytest"
uv pip install -q --python "$SIM" --no-deps -e "$DEV/robosuite" -e "$DEV/robocasa"
uv pip install -q --python "$SIM" -e "$REPO"
(cd "$DEV/robosuite" && "$SIM" robosuite/scripts/setup_macros.py >/dev/null) || true
(cd "$DEV/robocasa" && "$SIM" robocasa/scripts/setup_macros.py >/dev/null) || true

echo "== RoboCasa kitchen assets (~10 GB download)"
if [ ! -d "$DEV/robocasa/robocasa/models/assets/fixtures/fridges" ] || [ ! -d "$DEV/robocasa/robocasa/models/assets/objects/objaverse" ]; then
  (cd "$DEV/robocasa" && yes y | "$SIM" robocasa/scripts/download_kitchen_assets.py --type all) || true
fi
ls "$DEV/robocasa/robocasa/models/assets/fixtures/fridges" | wc -l | xargs echo "fridge models:"

echo "== skill venv (what a skill may import)"
[ -d "$LAB/venvs/skill" ] || uv venv -q --python 3.11 "$LAB/venvs/skill"
uv pip install -q --python "$LAB/venvs/skill/bin/python" numpy scipy opencv-python-headless pillow

echo "== vision venv (OWLv2, Grounding DINO, SAM 2 on CUDA)"
[ -d "$LAB/venvs/vision" ] || uv venv -q --python 3.11 "$LAB/venvs/vision"
uv pip install -q --python "$LAB/venvs/vision/bin/python" torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python "$LAB/venvs/vision/bin/python" transformers accelerate pillow numpy
HF_HOME="$LAB/hf" "$LAB/venvs/vision/bin/python" - <<'EOF'
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Owlv2Processor, Owlv2ForObjectDetection, Sam2Processor, Sam2Model
for ck in ["google/owlv2-base-patch16-ensemble"]:
    Owlv2Processor.from_pretrained(ck); Owlv2ForObjectDetection.from_pretrained(ck)
for ck in ["IDEA-Research/grounding-dino-tiny"]:
    AutoProcessor.from_pretrained(ck); AutoModelForZeroShotObjectDetection.from_pretrained(ck)
for ck in ["facebook/sam2.1-hiera-small"]:
    Sam2Processor.from_pretrained(ck); Sam2Model.from_pretrained(ck)
print("vision models cached")
EOF

echo "== rendering backend"
GL=egl
if ! MUJOCO_GL=egl PYOPENGL_PLATFORM=egl "$SIM" -c "import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\".1\"/></worldbody></mujoco>'); r=mujoco.Renderer(m); r.update_scene(mujoco.MjData(m)); r.render(); print('egl ok')"; then
  GL=osmesa; echo "EGL failed: falling back to OSMesa (CPU rendering, slower)"
fi
echo "export MUJOCO_GL=$GL PYOPENGL_PLATFORM=$GL" > "$LAB/env.sh"
cat >> "$LAB/env.sh" <<EOF
export FRIDGE_LAB_ROOT="$LAB"
export FRIDGE_SKILL_PYTHON="$LAB/venvs/skill/bin/python"
export FRIDGE_VISION_URL=http://127.0.0.1:8777
export FRIDGE_EXTRA_DENY="$DEV:$LAB/venvs/sim:$LAB/hf"
export HF_HOME="$LAB/hf"
export PATH="$HOME/.local/bin:\$PATH"
EOF

echo "== users: researcher (runs Codex), fridgeskill (runs skills)"
id researcher >/dev/null 2>&1 || useradd -m -s /bin/bash researcher
id fridgeskill >/dev/null 2>&1 || useradd -M -s /usr/sbin/nologin fridgeskill

echo "== permissions"
# Readable by everyone: the repo's code (the researcher may read the lab's source), the skill venv.
chmod 755 /opt "$LAB" "$LAB/venvs"
chmod -R a+rX "$LAB/venvs/skill" "$REPO"
# Not readable by the researcher or skills: keys, the scene splits and results, RoboCasa (test kitchens), the sim and
# vision venvs, model weights, the lab's private data and evals.
for p in "$REPO/.env" "$REPO/experiments/fridge_lab/scenes.json"; do [ -e "$p" ] && chmod 600 "$p"; done
for d in "$REPO/experiments/fridge_lab/results" "$REPO/experiments/astranomy/results" "$REPO/experiments/ortho_grasp/results" \
         "$DEV" "$LAB/venvs/sim" "$LAB/venvs/vision" "$LAB/hf" "$LAB/private" "$LAB/eval"; do
  mkdir -p "$d"; chmod 700 "$d"
done
chmod 700 /root

echo
echo "Setup done. Next (see experiments/fridge_lab/runpod/README.md):"
echo "  1. copy your .env to $REPO/.env (then: chmod 600 $REPO/.env)"
echo "  2. su - researcher -c 'codex login --device-auth'"
echo "  3. bash experiments/fridge_lab/runpod/check.sh"
