#!/usr/bin/env bash
set -euo pipefail
dep="$PWD/../as2w-r284-release/deliveries/as2w-deploy"
baseline="$(dirname "$0")/formal/model_800.pt"
out="$(dirname "$0")/support-aware-hip70-stairs-800"
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/code:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep" ../mujoco-venv/bin/python code/eval_as2w_mujoco_stepping.py assets/r237-train-root "$baseline" "$out" --seed 8302 --trials 4 --native-pd --leg-pd 70 2 --leg-pd-group hip --calf-mesh-diagnostic --down-approach .55 --sequence-period 8 --sequence-stride .35 --packet-loss .02 --target-margin .05 --sequence-attitude --predictive-guard --case up --case down
