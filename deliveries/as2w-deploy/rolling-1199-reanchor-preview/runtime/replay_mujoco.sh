#!/usr/bin/env bash
set -euo pipefail
pkg=$(cd -- "$(dirname -- "$0")/.." && pwd)
root=${AS2W_RUNTIME:?Set AS2W_RUNTIME to the AS2-W training runtime root}
dep="$root/../as2w-r284-release/deliveries/as2w-deploy"
out=${1:-"$PWD/rolling-1199-reanchor-results"}
exec env -u LD_LIBRARY_PATH PYTHONPATH="$pkg/runtime:$root/code:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep" \
  "$root/../mujoco-venv/bin/python" "$pkg/runtime/eval_entry_reanchor_holdout.py" \
  "$root/assets/r237-train-root" "$pkg/weights/model_1199.pt" "$out" \
  --measured-start --speed .05 --rolling-fixed-exit --rolling-stand-compensation \
  --rolling-wheel-hold --park-precalibrated --park-odometry --park-common-gain-scale .5 \
  --park-settle-joints --acceptance --packet-loss .02 --push-schedule all --native-pd \
  --substep-safety --trials 6 --park-continuous-bias --park-straight-bias
