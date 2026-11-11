set -euo pipefail
d=runs/r290-three-skills
smoke="$d/rolling-knee-margin-smoke"
smoke_pid=$(cat "$smoke/job.pid")
until grep -q AS2W_ROBUST_COMPLETE "$smoke/job.log"; do
  kill -0 "$smoke_pid" || exit 1
  sleep 5
done
limit=$(cat /sys/fs/cgroup/memory.max)
if test "$limit" != max; then
  while test "$((limit-$(cat /sys/fs/cgroup/memory.current)))" -lt 9663676416; do sleep 15; done
fi
model="$d/rolling-maneuvers-formal/model_1199.pt"
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
out="$d/rolling-knee-margin-formal"
mkdir "$out"
echo "$$" >"$out/job.pid"
exec /isaac-sim/python.sh code/train_as2w_isaac_robust.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs 384 --iterations 1200 --out "$out" --checkpoint "$model" --profile expanded --speed-cap 1 --speed-rolling --rolling-fixed-exit --rolling-stand-compensation --rolling-wheel-hold --park-precalibrated --park-continuous-bias --park-straight-bias --rolling-maneuvers --rolling-knee-margin --safety --learning-rate .00010 --seed 303 >"$out/job.log" 2>&1
