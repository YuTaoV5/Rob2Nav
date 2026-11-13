#!/usr/bin/env bash
set -euo pipefail
dep="$PWD/../as2w-r284-release/deliveries/as2w-deploy"
baseline=runs/stepping/support-aware-reward.bp4984/formal/model_1199.pt
out="$(dirname "$0")/mujoco-85-32-support-gate"
env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/code:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep" ../mujoco-venv/bin/python "$(dirname "$0")/eval-support-gate.py" assets/r237-train-root "$baseline" "$out" --seed 8302 --trials 4 --native-pd --leg-pd 85 3.2 --calf-mesh-diagnostic --down-approach .55 --sequence-period 8 --sequence-stride .35 --packet-loss .02 --target-margin .05 --sequence-attitude --predictive-guard --case up --case down
