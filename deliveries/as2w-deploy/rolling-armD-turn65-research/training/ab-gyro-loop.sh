#!/usr/bin/env bash
# A/B continuation from 4bdf4d26 (armE final) with gyro-1 INSIDE the training loop.
#
# Operator plan (2026-11-16): hot-start from 4bdf4d26, two 1,000-round arms, 4,096 envs, save every
# 100 rounds, screen every 200.  Change SAMPLING only - do not stack reward, gain or friction
# changes in the same round.
#
#   ARM=A  equal-budget control: existing sampling and reward, 1,000 more rounds.
#   ARM=B  long-turn curriculum: sampling and command dwell only (see the B block below).
#
# CONTROL CONTRACT (version gyro-1-trainloop): gyro-1 now lives in the training loop as well as in
# evaluation, because it previously existed only in the evaluation wrapper and every earlier policy
# was trained without it (see evidence/CONTROL_CONTRACT_GYRO_TRAIN_LOOP.md).  Both arms run the same
# loop, same gains.
#
# The previous recipe (armE-support2.sh) pinned init 92c44107 and robust_env 1af438db; both are
# superseded here, so this is a NEW recipe rather than an edit.
set -euo pipefail

ROOT=${ROOT:?set ROOT to the Rob2Nav checkout}
ARM=${ARM:?set ARM to A or B}
OUT=${OUT:?set OUT to the run directory}

BASE="$ROOT/training/as2w-backup/as2w-current"
TRAIN="$ROOT/skills/as2w/rolling/train_armE_support2p0.py"
INIT=${INIT:-/root/gpufree-data/as2w-runs/r11_fanziqi_lab/yaw-arms/armE-support-20261116-034514/r11-final.pt}

# ---- pin every input the run depends on -----------------------------------
printf '%s  %s\n' 4bdf4d26b076d7c717e9c151946dd42f8b3cf9dbbf592c5d6aac1ed5f6615866 "$INIT" | sha256sum -c -
printf '%s  %s
' a8b0e3f12da259e5c34a54235ec3aea15ad43a86ba2930e6e3e23954142e6bfd \
  "$BASE/code/as2w_isaac_robust_env.py" | sha256sum -c -
printf '%s  %s\n' e739dd1f9cd45ee1c280f49ffef1bde6db38719f0ad34e3b56d6f8b7bdd3b2ed \
  "$BASE/code/as2w_isaac_flat_env_all.py" | sha256sum -c -
printf '%s  %s\n' 8532a884c808b462dea70c8f48368bb768864b33cd0dc6177338e299570e0c9c \
  "$BASE/code/as2w_gyro_wheel_action.py" | sha256sum -c -
printf '%s  %s\n' a61b0bb57d41b32471efec16385bfddc33c358bd57aa47d262c7fce19ad1978a \
  "$TRAIN" | sha256sum -c -
printf '%s  %s\n' f969e454a1b4096eee2a52ddcb89fa1d6c63c977bdf6cfca10e3d1bdbfd51347 \
  "$BASE/code/as2w_ab_curriculum.py" | sha256sum -c -

# isaaclab + the AS2W code dir must be importable before handing over to python.sh
_pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${_pp}/opt/isaaclab-venv/lib/python3.12/site-packages:${BASE}/code"
unset LD_LIBRARY_PATH

export AS2W_R11_SAVE_INTERVAL=${AS2W_R11_SAVE_INTERVAL:-100}

# ---- gyro-1 inside the training loop (same gains as the evaluation wrapper) --
export AS2W_GYRO_KP=${AS2W_GYRO_KP:-60}
export AS2W_GYRO_CAP=${AS2W_GYRO_CAP:-4}
export AS2W_GYRO_FF_GAIN=${AS2W_GYRO_FF_GAIN:-0}

# ---- arm-specific sampling ------------------------------------------------
# The training entry reads these through the command manager; A keeps the armE values unchanged.
case "$ARM" in
  A)
    # equal-budget control: identical sampling and reward to armE
    ;;
  B)
    # long-turn curriculum: sampling and command dwell ONLY.
    #   35% pure turns  wz=+-0.05/+-0.10, vx=vy=0, dwell 8-12 s after the ramp
    #   25% arcs        vx=+0.25 with the frozen left/right yaw steps
    #   15% straight    low-speed continuous change plus +-0.25, vy=0
    #   15% brakes       forward and reverse in equal share, each reaching stop and fixed stand
    #   10% hold        pre-start hold and post-motion return to zero (re-captures wheel reference)
    export AS2W_AB_CURRICULUM=1   # this is what actually switches the sampling class
    export AS2W_AB_TURN_FRACTION=${AS2W_AB_TURN_FRACTION:-0.35}
    export AS2W_AB_ARC_FRACTION=${AS2W_AB_ARC_FRACTION:-0.25}
    export AS2W_AB_STRAIGHT_FRACTION=${AS2W_AB_STRAIGHT_FRACTION:-0.15}
    export AS2W_AB_BRAKE_FRACTION=${AS2W_AB_BRAKE_FRACTION:-0.15}
    export AS2W_AB_HOLD_FRACTION=${AS2W_AB_HOLD_FRACTION:-0.10}
    export AS2W_AB_TURN_DWELL_S=${AS2W_AB_TURN_DWELL_S:-10}
    export AS2W_AB_TURN_DWELL_MIN_S=${AS2W_AB_TURN_DWELL_MIN_S:-8}
    export AS2W_AB_TURN_DWELL_MAX_S=${AS2W_AB_TURN_DWELL_MAX_S:-12}
    ;;
  *) echo "ARM must be A or B" >&2; exit 2 ;;
esac

echo "AS2W_AB arm=$ARM init=$INIT gyro_kp=${AS2W_GYRO_KP} cap=${AS2W_GYRO_CAP} save=${AS2W_R11_SAVE_INTERVAL}"

AS2W_YAW_WEIGHT=3.0 AS2W_YAW_STD=0.05 AS2W_SYM_W=0.0 AS2W_SUPPORT_W=-2.0 \
  /isaac-sim/python.sh "$TRAIN" --viz none --device cuda:0 \
    --usd "$BASE/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda" \
    --urdf "$ROOT/training/as2w-backup/dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf" \
    --out "$OUT" --num-envs 4096 --iterations 1000 --init-checkpoint "$INIT"
