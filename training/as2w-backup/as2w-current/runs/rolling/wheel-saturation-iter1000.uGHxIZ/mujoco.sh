#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
dep="$PWD/../as2w-r284-release/deliveries/as2w-deploy"
export PYTHONPATH="$root/source:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep"
pids=()
for arm in baseline; do
    mkdir -p "$root/$arm"
    gain=0; if [[ $arm == rear ]]; then gain=.04; fi
    env -u LD_LIBRARY_PATH ../mujoco-venv/bin/python "$root/source/eval_heldout.py" assets/r237-train-root "/root/gpufree-data/Rob2Nav-runtime/as2w-current/runs/rolling/actor-wheel-saturation-loss.Uliv0s/baseline/train/model_1000.pt" "$root/$arm/mujoco" --speed .20 --rolling-fixed-exit --rolling-stand-compensation --rolling-wheel-hold --park-precalibrated --park-continuous-bias --park-straight-bias --pure-turn-common-scale 0 --park-policy-rad "$gain" --rolling-leg-filter-hz 5 --native-pd --substep-safety --acceptance --packet-loss .02 --trials 4 > "$root/$arm/mujoco.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
