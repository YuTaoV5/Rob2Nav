set -euo pipefail
d=runs/r290-three-skills
cpu="$d/switch-recalibrated-expanded-mujoco-r750-r200"
pid=${1:?missing MuJoCo diagnostic PID}
until test -s "$cpu/switch-contract.json"; do
  kill -0 "$pid" || exit 1
  sleep 10
done
limit=$(cat /sys/fs/cgroup/memory.max)
if test "$limit" != max; then
  while test "$((limit-$(cat /sys/fs/cgroup/memory.current)))" -lt 11811160064; do sleep 20; done
fi
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
mj_site=$(find "$PWD/../mujoco-venv/lib" -maxdepth 2 -type d -name site-packages -print -quit)
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
out="$d/switch-recalibrated-expanded-isaac-r750-r200"
mkdir "$out"
exec env PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages:$mj_site" /isaac-sim/python.sh code/eval_as2w_isaac_switch.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --rolling "$d/onnx-diagnostic/rolling-mirror-750" --stepping "$d/onnx-diagnostic/stepping-shaped-200" --out "$out" --native-position-pd --expanded --num-envs 4 --recalibrate-fixed-exit >"$out/job.log" 2>&1
