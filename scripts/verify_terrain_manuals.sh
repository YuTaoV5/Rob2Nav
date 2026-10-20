#!/usr/bin/env bash
# Execute the public terrain command manuals with fresh outputs; no hardware calls.
set -euo pipefail
cd /root/gpufree-data/Rob2Nav
export MOE="$PWD/vendor/RoboGauge/resources/models/go2/go2_moe_cts_137k_0.6739.pt"
export INIT="$PWD/artifacts/moe_memory_v2_end/training/visible_moe_memory4_v2_seed42/model_999.pt"
PY=/root/gpufree-data/envs/go2-mapless/bin/python
for skill in stairs platform multiterrain; do
  case "$skill" in
    stairs) kind=stairs;seed=71;kinds=stairs;;
    platform) kind=platform;seed=72;kinds=platform;;
    multiterrain) kind=mixed;seed=73;kinds='flat stairs platform';;
  esac
  "$PY" -m moe_perceptive.check --checkpoint "$MOE" --output "outputs/${skill}_model_check.json"
  /isaac-sim/python.sh -m moe_perceptive.train --headless --device cuda:0 \
    --moe "$MOE" --initialize-from "$INIT" --kind "$kind" --course original \
    --num-envs 512 --iterations 200 --warmup 20 --fusion 180 --actor 0 \
    --seed "$seed" --memory-lifetime 4 --sensor-rig original --numeric-debug \
    --studio-config "skills/$skill/studio_config.json" --output "outputs/${skill}_reproduction"
  test -f "outputs/${skill}_reproduction/completed.json"
  test -f "outputs/${skill}_reproduction/model_199.pt"
  for terrain in $kinds; do
    difficulty=.5
    if test "$terrain" = flat;then difficulty=0;fi
    /isaac-sim/python.sh -m moe_perceptive.evaluate --headless --device cuda:0 \
      --moe "$MOE" --checkpoint "outputs/${skill}_reproduction/model_199.pt" \
      --kind "$terrain" --difficulty "$difficulty" --num-envs 32 --seed 801 \
      --memory-lifetime 4 --sensor-rig original --course original \
      --output "outputs/${skill}_evaluation/${terrain}_seed801"
    test -f "outputs/${skill}_evaluation/${terrain}_seed801/summary.json"
  done
done
