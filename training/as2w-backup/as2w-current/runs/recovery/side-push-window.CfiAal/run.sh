#!/usr/bin/env bash
set -euo pipefail
out=$(dirname "$0")
dep="$PWD/../as2w-r284-release/deliveries/as2w-deploy"
model=runs/recovery/deep-side-curriculum/formal/model_1199.pt
bank=runs/r290-three-skills/supported-getup-holdout.json
for mode in original extended; do
  extra=()
  if test "$mode" = extended; then extra+=(--observe-after-push); fi
  env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/code:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep" ../mujoco-venv/bin/python code/eval_as2w_mujoco_recovery.py assets/r237-train-root "$model" "$bank" "$out/$mode" --seed 7301 --family left --family right --side-pose-bank runs/r290-three-skills/supported-getup-side-holdout.json --low-support-handoff --packet-loss .02 --predictive-guard "${extra[@]}"
done
