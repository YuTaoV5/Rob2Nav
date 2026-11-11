set -euo pipefail
d=runs/r290-three-skills
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
model="$d/stepping-go2-reset-exploration-init/model.pt"
busy=$(cat "$d/getup-reset-exploration-scheduler.pid")
while kill -0 "$busy" 2>/dev/null; do sleep 15; done
envs=256
budget=201
out="$d/stepping-go2-reset-exploration-stairs-formal"
  mkdir "$out"
  echo "$$" >"$out/job.pid"
  /isaac-sim/python.sh code/train_as2w_isaac_stepping.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs "$envs" --iterations "$budget" --out "$out" --init-checkpoint "$model" --curriculum stairs-sequence-guarded --sequence-period 8 --sequence-stride .35 --sequence-shaped-reward --directed-stair-curriculum --packet-loss .02 --target-margin .05 --sequence-attitude --predictive-guard --privileged-stair-critic --go2-support-reference "$d/go2-as2w-wheel-teacher-projected" --seed 305 >"$out/job.log" 2>&1
