#!/usr/bin/env bash
set -euo pipefail
out="$(dirname "$0")/formal"
mkdir -p "$out"
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
model="runs/r290-three-skills/stepping-terrain-critic-stairs-formal/model_1199.pt"
exec /isaac-sim/python.sh code/train_as2w_isaac_stepping.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs 256 --iterations 1200 --out "$out" --init-checkpoint "$model" --curriculum stairs-sequence-guarded --sequence-period 8 --sequence-stride .35 --sequence-shaped-reward --directed-stair-curriculum --packet-loss .02 --target-margin .05 --sequence-attitude --predictive-guard --privileged-stair-critic --support-aware-swing --seed 305
