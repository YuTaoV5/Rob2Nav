#!/usr/bin/env bash
# armE: single-variable continuation from model_3998 with rolling_wheel_support -0.65 -> -2.0
#
# Measured result (MuJoCo 11-case matrix, 5 settle states): 7/11 at the endpoint, both arcs
# PASS with four-wheel support >= 0.987.  On Isaac the same weights score 5/11 == baseline.
# Research status; see deliveries/as2w-deploy/rolling-r11-turn-gyro-research/.
#
# The only variable versus the armA baseline is the support weight.  Learning rate is the
# inherited adaptive 3e-4 (PPO.load restores it from the checkpoint, so an env override set
# before load has no effect - see the package runtime/README.md).
set -euo pipefail

ROOT=${ROOT:?set ROOT to the Rob2Nav checkout}
BASE="$ROOT/training/as2w-backup/as2w-current"
TRAIN="$ROOT/skills/as2w/rolling/train_armE_support2p0.py"
INIT=${INIT:?set INIT to model_3998.pt}
OUT=${OUT:?set OUT to the run directory}
export AS2W_R11_SAVE_INTERVAL=${AS2W_R11_SAVE_INTERVAL:-100}

# fail fast on the two inputs that must be pinned
printf '%s  %s\n' 92c44107bc16e4783e8b8cacc6c1d012f3f86e6700860a6a85badedbe86dfd66 "$INIT" | sha256sum -c -
printf '%s  %s\n' 1af438db2ff3babc064a3a56bc8825880a5d3640b2d0f76174b7faa35233e24f \
  "$BASE/code/as2w_isaac_robust_env.py" | sha256sum -c -

AS2W_YAW_WEIGHT=3.0 AS2W_YAW_STD=0.05 AS2W_SYM_W=0.0 AS2W_SUPPORT_W=-2.0 \
  /isaac-sim/python.sh "$TRAIN" --viz none --device cuda:0 \
    --usd "$BASE/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda" \
    --urdf "$ROOT/training/as2w-backup/dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf" \
    --out "$OUT" --num-envs 4096 --iterations 600 --init-checkpoint "$INIT"
