set -euo pipefail
out="$(dirname "$0")/isaac-85-32"
mkdir -p "$out"
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
model=runs/stepping/support-aware-reward.bp4984/formal/model_1199.pt
exec /isaac-sim/python.sh code/train_as2w_isaac_stepping.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs 40 --iterations 1 --out "$out" --eval-checkpoint "$model" --eval-leg-pd 85 3.2 --curriculum stairs-sequence-guarded --sequence-shaped-reward --seed 6302 --sequence-period 8 --sequence-stride .35 --packet-loss .02 --target-margin .05 --sequence-attitude --predictive-guard --directed-stair-curriculum
