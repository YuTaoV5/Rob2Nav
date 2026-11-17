#!/usr/bin/env bash
# fs_stair_6frame.sh -- H6 arm (six consecutive frames, 360 dim, 100 ms span).
#
#   dry run (default):   ./fs_stair_6frame.sh
#   train (authorized):  ./fs_stair_6frame.sh --run --out <evidence dir>
#
# `env -u LD_LIBRARY_PATH` and torch.set_num_threads(1) (inside
# fs_stair_train.py) are mandatory for reproducibility.
set -euo pipefail
cd "$(dirname "$0")"
exec env -u LD_LIBRARY_PATH /opt/isaaclab-venv/bin/python fs_stair_train.py \
  --arch 6frame "$@"
