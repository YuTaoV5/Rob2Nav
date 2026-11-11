set -euo pipefail
d=runs/r290-three-skills
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
model="$d/stepping-go2-reset-exploration-init/model.pt"
smoke="$d/stepping-stance-margin-v2-smoke"
until grep -q AS2W_STEPPING_COMPLETE "$smoke/job.log"; do
  kill -0 "$(cat "$smoke/job.pid")" || exit 1
  sleep 5
done
grep -q 'stepping_stance_margin' "$smoke/job.log"
grep -q 'stepping_knee_margin' "$smoke/job.log"
limit=$(cat /sys/fs/cgroup/memory.max)
if test "$limit" != max; then
  while test "$((limit-$(cat /sys/fs/cgroup/memory.current)))" -lt 9663676416; do sleep 15; done
fi
envs=256
budget=1200
out="$d/stepping-calf-clearance-formal"
  mkdir "$out"
  echo "$$" >"$out/job.pid"
  /isaac-sim/python.sh code/train_as2w_isaac_stepping.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs "$envs" --iterations "$budget" --out "$out" --init-checkpoint "$model" --curriculum stairs-sequence-guarded --sequence-period 8 --sequence-stride .35 --sequence-shaped-reward --directed-stair-curriculum --packet-loss .02 --target-margin .05 --stance-margin-reward --calf-clearance-reward --sequence-attitude --predictive-guard --privileged-stair-critic --go2-support-reference "$d/go2-as2w-wheel-teacher-projected" --seed 305 >"$out/job.log" 2>&1
