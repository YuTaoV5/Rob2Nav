set -euo pipefail
d=runs/r290-three-skills
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
mj_site=$(find "$PWD/../mujoco-venv/lib" -maxdepth 2 -type d -name site-packages -print -quit)
dep="$PWD/../as2w-r284-release/deliveries/as2w-deploy"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
rolling="$d/onnx-diagnostic/rolling-mirror-750"
stepping="$d/onnx-diagnostic/stepping-shaped-200"
limit=$(cat /sys/fs/cgroup/memory.max)
for order in rolling-first stepping-first; do
  extra=()
  if test "$order" = stepping-first; then extra=(--reverse-order); fi
  out="$d/switch-substep-mujoco-$order"
  mkdir "$out"
  env PYTHONPATH="$PWD/code:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep" ../mujoco-venv/bin/python code/eval_as2w_mujoco_switch.py assets/r237-train-root "$rolling" "$stepping" "$out" --trials 4 --expanded --fade-stand-bias --predictive-guard "${extra[@]}" >"$out/job.log" 2>&1
done
for order in rolling-first stepping-first; do
  extra=()
  if test "$order" = stepping-first; then extra=(--reverse-order); fi
  if test "$limit" != max; then
    while test "$((limit-$(cat /sys/fs/cgroup/memory.current)))" -lt 11811160064; do sleep 20; done
  fi
  out="$d/switch-substep-isaac-$order"
  mkdir "$out"
  env PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages:$mj_site" /isaac-sim/python.sh code/eval_as2w_isaac_switch.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --rolling "$rolling" --stepping "$stepping" --out "$out" --native-position-pd --expanded --num-envs 4 "${extra[@]}" >"$out/job.log" 2>&1
done
