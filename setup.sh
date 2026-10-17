#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON_BIN="${GO2_PYTHON:-python3.11}"
"$PYTHON_BIN" scripts/check_vendor.py
"$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" -m pip freeze > environment.lock.txt
MUJOCO_GL="${MUJOCO_GL:-egl}" "$PYTHON_BIN" -m unittest discover -s tests -v
