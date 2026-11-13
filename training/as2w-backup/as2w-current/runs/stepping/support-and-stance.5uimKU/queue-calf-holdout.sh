#!/usr/bin/env bash
set -euo pipefail
out=$(dirname "$0")
pid=$(cat "$out/calf-diagnostic.pid")
until test -f "$out/calf-diagnostic.exit"; do
  kill -0 "$pid"
  sleep 5
done
test "$(cat "$out/calf-diagnostic.exit")" = 0
unzip -tq "$out/calf-diagnostic/model_24.pt" >/dev/null
exec bash "$out/run-holdout.sh" calf-diagnostic
