#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
init="$PWD/runs/r290-three-skills/rolling-knee-margin-formal/model_1199.pt"
export PYTHONPATH="$root/source:${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
common=(--viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs 24 --profile expanded --speed-cap .20 --speed-rolling --rolling-fixed-exit --rolling-stand-compensation --rolling-wheel-hold --park-precalibrated --park-continuous-bias --park-straight-bias --pure-turn-common-scale 0 --rolling-leg-filter-hz 5 --rolling-low-speed-smooth --safety)
for arm in baseline; do
    gain=0; if [[ $arm == rear ]]; then gain=.04; fi
    mkdir -p "$root/$arm"/{train,isaac}
    /isaac-sim/python.sh "$root/source/train_as2w_isaac_robust.py" "${common[@]}" --park-policy-rad "$gain" --iterations 1 --out "$root/$arm/isaac" --checkpoint "/root/gpufree-data/Rob2Nav-runtime/as2w-current/runs/rolling/actor-wheel-saturation-loss.Uliv0s/baseline/train/model_1000.pt" --evaluate --seed 7393 > "$root/$arm/isaac/job.log" 2>&1
done
