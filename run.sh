#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [ -n "${GO2_PYTHON:-}" ]; then
  PYTHON_BIN="$GO2_PYTHON"
elif [ -x .venv/bin/python ]; then
  PYTHON_BIN="$PWD/.venv/bin/python"
elif [ -x /root/gpufree-data/envs/go2-mapless/bin/python ]; then
  PYTHON_BIN=/root/gpufree-data/envs/go2-mapless/bin/python
else
  echo "Set GO2_PYTHON to the Python 3.11 environment containing the project dependencies."
  exit 1
fi
exec "$PYTHON_BIN" "$@"
