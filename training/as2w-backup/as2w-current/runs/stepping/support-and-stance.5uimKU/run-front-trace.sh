#!/usr/bin/env bash
set -euo pipefail
dep="$PWD/../as2w-r284-release/deliveries/as2w-deploy"
out_script=$(dirname "$0")
baseline="$(dirname "$0")/$1/model_24.pt"
out="$(dirname "$0")/front-trace-$1"
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/code:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep" ../mujoco-venv/bin/python "$out_script/eval-front-trace.py" assets/r237-train-root "$baseline" "$out" --seed 8302 --trials 1 --native-pd --leg-pd 55 2 --calf-mesh-diagnostic --down-approach .55 --sequence-period 8 --sequence-stride .35 --packet-loss .02 --target-margin .05 --sequence-attitude --predictive-guard --rear-stance-predictor --case up --case down
