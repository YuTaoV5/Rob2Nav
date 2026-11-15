#!/usr/bin/env bash
# AS2-W rolling · turn-biased continuation (2000 iterations) from model_1999,
# with the action-delay domain randomisation DISABLED.
#
# Usage: bash skills/as2w/rolling/recipes/turn-2000fix.sh
# Env:   AS2W_R11_ENVS (default 4096), AS2W_R11_ITERS (default 2000),
#        AS2W_R11_SAVE_INTERVAL (default 100), AS2W_R11_OUT (default under runs/)
#
# Only the command mix changes vs the parent config:
#   hold 0.20 / straight 0.15 / spin 0.40 / arc 0.25
# Rewards, wheel drive (kp=0, kd=6, +-13.5 Nm), leg gains 55/2 and the 90% hard
# protection are inherited unchanged.
#
# The recipe verifies BOTH hashes and fails closed:
#   init checkpoint  f54751faddc211b4cfa7498ac35e95e4f2617779bcdb8b3b5de5904c9da72985
#   env source       1af438db2ff3babc064a3a56bc8825880a5d3640b2d0f76174b7faa35233e24f
# The second hash is the action-delay-disable patch. If your checkout still has the
# historical source (fa648f7e...), apply
#   deliveries/as2w-deploy/rolling-r11-turn-research/runtime/action-delay-disable.patch
# first, or set as2w_action_delay_max_steps=0 through the config.
set -euo pipefail

root="$(cd "$(dirname "$0")/../../../.." && pwd)"
base="$root/training/as2w-backup/as2w-current"
here="$(cd "$(dirname "$0")/.." && pwd)"

init=${AS2W_R11_INIT:-/root/gpufree-data/as2w-runs/r11_fanziqi_lab/resume-20261115-204347-8722/model_1999.pt}
want_init=f54751faddc211b4cfa7498ac35e95e4f2617779bcdb8b3b5de5904c9da72985
env_py="$base/code/as2w_isaac_robust_env.py"
want_env=1af438db2ff3babc064a3a56bc8825880a5d3640b2d0f76174b7faa35233e24f

printf '%s  %s\n' "$want_init" "$init"  | sha256sum -c -
printf '%s  %s\n' "$want_env"  "$env_py" | sha256sum -c -

envs=${AS2W_R11_ENVS:-4096}
iters=${AS2W_R11_ITERS:-2000}
export AS2W_R11_SAVE_INTERVAL=${AS2W_R11_SAVE_INTERVAL:-100}

out=${AS2W_R11_OUT:-$root/runs/r11-turn-$(date +%Y%m%d-%H%M%S)}
usd="$base/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$root/training/as2w-backup/dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
pp=$(printf "%s:" /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="$base/code:${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
unset LD_LIBRARY_PATH
cd "$base"

mkdir -p "$out"
git -C "$root" rev-parse HEAD > "$out/source-revision.txt" || true

/isaac-sim/python.sh "$here/train_r11.py" \
  --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" \
  --out "$out" --num-envs "$envs" --iterations "$iters" \
  --init-checkpoint "$init" 2>&1 | tee "$out/train.log"
