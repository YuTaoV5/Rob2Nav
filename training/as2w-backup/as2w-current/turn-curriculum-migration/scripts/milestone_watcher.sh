#!/bin/bash
# Durable milestone watcher for the A arm.
# Instead of a human/agent polling every few minutes, this process lives on the server and, whenever the A
# arm's training log reaches a milestone checkpoint (1000 / 2000 / 3000), runs the SANCTIONED evaluation entry
# on that checkpoint for the two held-out seeds and drops the output where stage2_compare.py expects it.
# It never touches the training process; it only reads the log and starts short 24-env evaluations.
set -u
R=/root/gpufree-data/Rob2Nav
BASE=$R/training/as2w-backup/as2w-current
RD=$BASE/runs/rolling/filtered-4000-env4096-stage020.ovvRL3
SNAP=$RD/source
T=$BASE/turn-curriculum-migration
W=$T/work
ARM_DIR=$W/A_nominal_r4000_curriculum
LOG=$ARM_DIR/job.log
USB="$BASE/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
URDF=$R/training/as2w-backup/dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf
OUTLOG=$W/AB-run/milestones.log
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages:$SNAP"
unset LD_LIBRARY_PATH
COMMON="--viz none --device cuda:0 --usd $USB --urdf $URDF --profile expanded --speed-cap .20 --speed-rolling --rolling-fixed-exit --rolling-stand-compensation --rolling-wheel-hold --park-precalibrated --park-continuous-bias --park-straight-bias --pure-turn-common-scale 0 --rolling-leg-filter-hz 5 --rolling-low-speed-smooth --safety --park-policy-rad 0"
echo "WATCHER_START $(date -Iseconds)" >> "$OUTLOG"
cd "$BASE"
while true; do
  IT=$(grep -aoE 'Learning iteration [0-9]+' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+' || echo 0)
  for M in 1000 2000 3000; do
    [ "$IT" -ge "$M" ] || continue
    CK=$ARM_DIR/model_$M.pt
    [ -f "$CK" ] || continue
    MARK=$W/AB-run/done_A_$M
    [ -f "$MARK" ] && continue
    echo "MILESTONE A=$M reached at $(date -Iseconds) ckpt=$(stat -c%s "$CK")B" >> "$OUTLOG"
    for SEED in 7393 1849; do
      D=$W/mideval_A${M}_seed$SEED
      mkdir -p "$D"
      /isaac-sim/python.sh "$SNAP/train_as2w_isaac_robust.py" $COMMON --num-envs 24 \
        --iterations 1 --out "$D" --checkpoint "$CK" --evaluate --seed "$SEED" > "$D/job.log" 2>&1
      echo "EVAL A=$M seed=$SEED rc=$? $(date -Iseconds)" >> "$OUTLOG"
    done
    touch "$MARK"
  done
  if ! ps -p "$(cat /root/_chainpid 2>/dev/null)" >/dev/null 2>&1; then
    echo "CHAIN_GONE at $(date -Iseconds) it=$IT" >> "$OUTLOG"
    sleep 300
  fi
  sleep 120
done
