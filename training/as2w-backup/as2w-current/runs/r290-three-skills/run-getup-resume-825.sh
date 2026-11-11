set -euo pipefail
d=runs/r290-three-skills
run="$d/getup-symmetric-formal"
model="$run/model_825.pt"
unzip -tq "$model"
if kill -0 "$(cat "$run/job.pid")" 2>/dev/null; then exit 1; fi
limit=$(cat /sys/fs/cgroup/memory.max)
if test "$limit" != max; then
  while test "$((limit-$(cat /sys/fs/cgroup/memory.current)))" -lt 8589934592; do sleep 20; done
fi
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
echo "$$" >"$run/job.pid"
exec env PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages" /isaac-sim/python.sh code/train_as2w_isaac_getup.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs 256 --iterations 1200 --out "$run" --resume-checkpoint "$model" --seed 297 --pose-bank "$d/supported-getup-poses.json" --staged-rise --rise-gain 85 --support-handoff --fixed-stand-integral --low-support-handoff --recovery-symmetry >"$run/resume-825.log" 2>&1
