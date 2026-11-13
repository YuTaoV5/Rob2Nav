set -euo pipefail
d=runs/r290-three-skills
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
out="$(dirname "$0")/$1"
mkdir -p "$out"
extra=()
if test "$1" = reward; then extra=(--bilateral-support-reward); fi
model="runs/recovery/deep-side-curriculum/formal/model_1199.pt"
exec /isaac-sim/python.sh code/train_as2w_isaac_getup.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs 128 --iterations 25 --out "$out" --init-checkpoint "$model" --seed 306 --pose-bank "$d/supported-getup-poses.json" --side-pose-bank runs/recovery/deep-side-curriculum/training-poses.json --side-pose-probability .65 --staged-rise --rise-gain 85 --support-handoff --fixed-stand-integral --low-support-handoff --recovery-symmetry --packet-loss .02 --predictive-guard --settled-stand-reward "${extra[@]}"
