#!/usr/bin/env bash
# fs_stair_1frame.sh -- S arm (single frame, 60 dim), one copy-paste command.
#
#   dry run (default, prints the pre-flight report and exits 0):
#       ./fs_stair_1frame.sh
#   train (only with the operator's explicit authorization):
#       ./fs_stair_1frame.sh --run --out /root/gpufree-data/as2w-runs/w1w2-stair-scratch.kuZUnA/fs_stair_s_run
#
# The interpreter MUST run with LD_LIBRARY_PATH cleared: `env -u LD_LIBRARY_PATH`
# is the only reason this wrapper exists (torch/mujoco otherwise fail with
# `undefined symbol: _PyErr_SetLocaleString`).
# torch.set_num_threads(1) is enforced inside fs_stair_train.py (the dry-run
# report prints the exact line as source.torch_threads.enforced_at); without it
# the same seed produces a different trajectory.
set -euo pipefail
cd "$(dirname "$0")"
exec env -u LD_LIBRARY_PATH /opt/isaaclab-venv/bin/python fs_stair_train.py \
  --arch 1frame "$@"
