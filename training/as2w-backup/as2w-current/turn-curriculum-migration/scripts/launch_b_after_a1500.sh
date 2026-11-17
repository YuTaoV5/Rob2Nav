#!/bin/bash
set -u
R=/root/gpufree-data/Rob2Nav
BASE=$R/training/as2w-backup/as2w-current
RD=$BASE/runs/rolling/filtered-4000-env4096-stage020.ovvRL3
SNAP=$RD/source
T=$BASE/turn-curriculum-migration
W=$T/work
A=$W/A_nominal_r4000_curriculum
LOG=$W/AB-run/b_launch.log
say(){ echo "$(date -Iseconds) $*" >> "$LOG"; }
say WAIT_START
while [ ! -f "$A/model_1500.pt" ]; do sleep 15; done
say A_1500_CHECKPOINT_PRESENT size=$(stat -c%s "$A/model_1500.pt")
sleep 10
for i in $(seq 1 120); do
  [ -f "$W/AB-run/done_A_1500" ] && { say A_1500_EVAL_DONE; break; }
  sleep 15
done
say STOPPING_A
for i in 1 2 3; do
  pkill -TERM -f "A_nominal_r4000_curriculum" 2>/dev/null || true
  sleep 8
  pgrep -f "A_nominal_r4000_curriculum" >/dev/null || break
done
pkill -TERM -f "ab_chain.sh" 2>/dev/null || true
sleep 10
say A_STOPPED remaining=$(pgrep -fc 'kit/python/bin/python3' || echo 0)
nvidia-smi --query-gpu=memory.used --format=csv,noheader >> "$LOG"
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages:$T/scripts:$SNAP"
export AS2W_R4000_SNAPSHOT="$SNAP" AS2W_AB_ARM=B AS2W_AB_NUM_STEPS=128 AS2W_TURN_B_SEED=311
unset LD_LIBRARY_PATH
cd "$BASE"
USB="$BASE/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
URDF="$R/training/as2w-backup/dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
INIT="$R/deliveries/as2w-deploy/rolling-4000-stage020-research/weights/model_3999.pt"
say LAUNCHING_B_1500
/isaac-sim/python.sh "$T/scripts/train_ab_entry.py" --viz none --device cuda:0 \
  --usd "$USB" --urdf "$URDF" --profile expanded --speed-cap .20 --speed-rolling \
  --rolling-fixed-exit --rolling-stand-compensation --rolling-wheel-hold \
  --park-precalibrated --park-continuous-bias --park-straight-bias \
  --pure-turn-common-scale 0 --rolling-leg-filter-hz 5 --rolling-low-speed-smooth \
  --safety --park-policy-rad 0 --rolling-maneuvers --rolling-knee-margin \
  --num-envs 16000 --iterations 1500 --out "$W/B_turn_curriculum_migrated" \
  --checkpoint "$INIT" --seed 311 --learning-rate .00005 > "$W/B_turn_curriculum_migrated/job.log" 2>&1
say B_EXIT=$?
