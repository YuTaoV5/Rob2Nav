#!/bin/bash
# Operator 09:29: start curriculum B now instead of waiting for A=1500, but evaluate the A endpoint first.
# Equal-budget is preserved by taking A's last saved checkpoint (model_N.pt) as the A endpoint and running B for
# exactly N iterations from the same init, with the same contract, seed and env count.
set -u
R=/root/gpufree-data/Rob2Nav
BASE=$R/training/as2w-backup/as2w-current
RD=$BASE/runs/rolling/filtered-4000-env4096-stage020.ovvRL3
SNAP=$RD/source
T=$BASE/turn-curriculum-migration
W=$T/work
A=$W/A_nominal_r4000_curriculum
B=$W/B_turn_curriculum_migrated
LOG=$W/AB-run/switch_to_b.log
USB="$BASE/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
URDF="$R/training/as2w-backup/dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
INIT="$R/deliveries/as2w-deploy/rolling-4000-stage020-research/weights/model_3999.pt"
say(){ echo "$(date -Iseconds) $*" >> "$LOG"; }

# 1) retire the waiting switcher and the original chain: we are taking manual control
pkill -f launch_b_after_a1500.sh 2>/dev/null || true
pkill -f ab_chain.sh 2>/dev/null || true
sleep 3

# 2) choose the A endpoint: latest saved checkpoint, iteration encoded in its name
CK=$(ls -t "$A"/model_*.pt 2>/dev/null | head -1)
if [ -z "${CK:-}" ]; then say NO_A_CHECKPOINT; exit 1; fi
N=$(basename "$CK" .pt | sed 's/model_//')
say A_ENDPOINT $CK iter=$N size=$(stat -c%s "$CK")

# 3) stop A precisely by its own output path (cannot touch anything else)
for i in 1 2 3 4; do
  pkill -TERM -f "A_nominal_r4000_curriculum" 2>/dev/null || true
  sleep 6
  pgrep -f "A_nominal_r4000_curriculum" >/dev/null || break
done
sleep 6
LEFT=$(pgrep -fc 'kit/python/bin/python3' 2>/dev/null || echo 0)
say A_STOPPED leftover_kit_python=$LEFT
if [ "${LEFT:-0}" -gt 0 ]; then
  # only kill what is still running the A output path
  for p in $(pgrep -f 'kit/python/bin/python3' 2>/dev/null || true); do
    grep -q "A_nominal_r4000_curriculum" /proc/$p/cmdline 2>/dev/null && kill -TERM "$p" 2>/dev/null || true
  done
  sleep 8
fi
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> "$LOG"

# 4) evaluate the A endpoint on both held-out seeds BEFORE starting B (operator said evaluate first)
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages:$SNAP"
unset LD_LIBRARY_PATH
COMMON="--viz none --device cuda:0 --usd $USB --urdf $URDF --profile expanded --speed-cap .20 --speed-rolling --rolling-fixed-exit --rolling-stand-compensation --rolling-wheel-hold --park-precalibrated --park-continuous-bias --park-straight-bias --pure-turn-common-scale 0 --rolling-leg-filter-hz 5 --rolling-low-speed-smooth --safety --park-policy-rad 0"
cd "$BASE"
for SEED in 7393 1849; do
  D=$W/finalA_seed$SEED
  rm -rf "$D"; mkdir -p "$D"
  say FINAL_A_EVAL_START seed=$SEED ckpt=$CK
  /isaac-sim/python.sh "$SNAP/train_as2w_isaac_robust.py" $COMMON --num-envs 24 \
    --iterations 1 --out "$D" --checkpoint "$CK" --evaluate --seed "$SEED" > "$D/job.log" 2>&1
  say FINAL_A_EVAL_DONE seed=$SEED rc=$?
done

# 5) start B for exactly N iterations (equal budget), same init/contract/seed
rm -rf "$B"; mkdir -p "$B"
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages:$T/scripts:$SNAP"
export AS2W_R4000_SNAPSHOT="$SNAP" AS2W_AB_ARM=B AS2W_AB_NUM_STEPS=128 AS2W_TURN_B_SEED=311
say LAUNCH_B iterations=$N
/isaac-sim/python.sh "$T/scripts/train_ab_entry.py" --viz none --device cuda:0 \
  --usd "$USB" --urdf "$URDF" --profile expanded --speed-cap .20 --speed-rolling \
  --rolling-fixed-exit --rolling-stand-compensation --rolling-wheel-hold \
  --park-precalibrated --park-continuous-bias --park-straight-bias \
  --pure-turn-common-scale 0 --rolling-leg-filter-hz 5 --rolling-low-speed-smooth \
  --safety --park-policy-rad 0 --rolling-maneuvers --rolling-knee-margin \
  --num-envs 16000 --iterations "$N" --out "$B" --checkpoint "$INIT" --seed 311 \
  --learning-rate .00005 > "$B/job.log" 2>&1
say B_EXIT=$?
