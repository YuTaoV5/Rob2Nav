#!/usr/bin/env bash
# fs_roll_C.sh - arm C rolling-from-scratch launcher.
# DEFAULT IS A DRY RUN: pure CPU python, no isaaclab import, no stage, no CUDA.
# Training starts ONLY with an explicit --run (operator instruction required).
set -uo pipefail
RUN_DIR=/root/gpufree-data/as2w-runs/fs-roll-20261116-2345
A2=/root/gpufree-data/Rob2Nav/training/as2w-backup/as2w-current
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages:${A2}/code"
unset LD_LIBRARY_PATH
cd "$RUN_DIR"
RUN_TRAIN=0
for arg in "$@"; do
  if [ "$arg" = "--run" ]; then RUN_TRAIN=1; fi
done
if [ "$RUN_TRAIN" = "1" ]; then
  echo "[fs_roll_C] --run given: TRAINING (Isaac/Kit on GPU). Operator-only path."
  exec /isaac-sim/python.sh "$RUN_DIR/fs_roll_train.py" --arm C --kd 1.5 \
       --config "$RUN_DIR/configs/fs_roll_C.json" "$@"
fi
echo "[fs_roll_C] DEFAULT dry-run: no Isaac, no GPU, no weights. Pass --run to train."
/usr/bin/python3 "$RUN_DIR/fs_roll_train.py" --arm C --kd 1.5 \
    --config "$RUN_DIR/configs/fs_roll_C.json" "$@" 2>&1 | tee -a "$RUN_DIR/dryrun.log"
exit "${PIPESTATUS[0]}"

