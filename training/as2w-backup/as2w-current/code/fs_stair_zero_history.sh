#!/usr/bin/env bash
# fs_stair_zero_history.sh -- H0 arm (360 dim, historical slots pinned to zero).
# Same shape and parameter count as H6; only the history content differs.
#
#   dry run (default):   ./fs_stair_zero_history.sh
#   train (authorized):  ./fs_stair_zero_history.sh --run --out <evidence dir>
#
# `env -u LD_LIBRARY_PATH` and torch.set_num_threads(1) (inside
# fs_stair_train.py) are mandatory for reproducibility.
set -euo pipefail
cd "$(dirname "$0")"
exec env -u LD_LIBRARY_PATH /opt/isaaclab-venv/bin/python fs_stair_train.py \
  --arch zero_history "$@"
