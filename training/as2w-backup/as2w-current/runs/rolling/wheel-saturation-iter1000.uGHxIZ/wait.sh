#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
training=$1
sim=$2
while ! unzip -tq "$training/baseline/train/model_1000.pt" >/dev/null 2>&1; do
    kill -0 "$(cat "$training/pid")"
    sleep 10
done
exec bash "$root/$sim.sh"
