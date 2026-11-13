#!/usr/bin/env bash
set -euo pipefail
dep="$PWD/../as2w-r284-release/deliveries/as2w-deploy"
baseline=runs/r290-three-skills/stepping-go2-support-stairs-formal/model_200.pt
out="$(dirname "$0")/go2-baseline-matched-stairs-200"
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/code:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep" ../mujoco-venv/bin/python code/eval_as2w_mujoco_stepping.py assets/r237-train-root "$baseline" "$out" --seed 8302 --trials 4 --native-pd --leg-pd 55 2 --calf-mesh-diagnostic --down-approach .55 --sequence-period 8 --sequence-stride .35 --packet-loss .02 --target-margin .05 --sequence-attitude --predictive-guard --case up --case down
