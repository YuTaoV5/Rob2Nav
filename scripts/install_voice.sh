#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
"$PYTHON_BIN" -m venv .voice-venv
.voice-venv/bin/python -m pip install -r requirements-voice.txt
.voice-venv/bin/python scripts/download_voice_models.py
printf '%s\n' 'Voice environment ready. Start: .voice-venv/bin/python -m voice_agent --mode demo --speech-config configs/speech.local.json'
