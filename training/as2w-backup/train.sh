#!/usr/bin/env bash
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
cd "$base/as2w-current"
[[ $# == 2 ]] || { echo "Usage: bash train.sh {rolling|stepping|recovery} NEW_OUTPUT_DIRECTORY" >&2; exit 2; }
d=runs/r290-three-skills
sim=${ISAAC_SIM_PYTHON:-/isaac-sim/python.sh}
pp=$(printf '%s:' "${ISAACLAB_SOURCE:-/workspace/isaaclab/source}"/isaaclab*)
export PYTHONPATH="${pp}${ISAACLAB_SITE_PACKAGES:-/opt/isaaclab-venv/lib/python3.12/site-packages}"
unset LD_LIBRARY_PATH
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
case "$1" in
 rolling)
  trainer=train_as2w_isaac_robust.py; count=384
  init="$d/rolling-maneuvers-formal/model_1199.pt"
  args=(--checkpoint "$init" --profile expanded --speed-cap 1 --speed-rolling --rolling-fixed-exit --rolling-stand-compensation --rolling-wheel-hold --park-precalibrated --park-continuous-bias --park-straight-bias --rolling-maneuvers --rolling-knee-margin --safety --learning-rate .00010 --seed 303)
  ;;
 stepping)
  trainer=train_as2w_isaac_stepping.py; count=256
  init="$d/stepping-go2-reset-exploration-init/model.pt"
  args=(--init-checkpoint "$init" --curriculum stairs-sequence-guarded --sequence-period 8 --sequence-stride .35 --sequence-shaped-reward --directed-stair-curriculum --packet-loss .02 --target-margin .05 --stance-margin-reward --sequence-attitude --predictive-guard --privileged-stair-critic --go2-support-reference "$d/go2-as2w-wheel-teacher-projected" --seed 305)
  ;;
 recovery)
  trainer=train_as2w_isaac_getup.py; count=384
  init="$d/getup-symmetric-formal/model_1199.pt"
  args=(--init-checkpoint "$init" --seed 301 --pose-bank "$d/supported-getup-poses.json" --side-pose-bank "$d/getup-training-mixed-side.json" --side-pose-probability .65 --staged-rise --rise-gain 85 --support-handoff --fixed-stand-integral --low-support-handoff --recovery-symmetry --packet-loss .02 --predictive-guard)
  ;;
 *) echo "Unknown task: $1" >&2; exit 2 ;;
esac
for file in "$usd" "$urdf" "$init"; do
  [[ -s "$file" ]] || { echo "Missing input: $file" >&2; exit 1; }
done
[[ -x "$sim" ]] || { echo "Missing Isaac Sim interpreter: $sim" >&2; exit 1; }
mkdir -- "$2"
exec "$sim" "code/$trainer" --viz none --device "${AS2W_DEVICE:-cuda:0}" --usd "$usd" --urdf "$urdf" --num-envs "${AS2W_NUM_ENVS:-$count}" --iterations "${AS2W_ITERATIONS:-1200}" --out "$2" "${args[@]}"
