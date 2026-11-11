set -euo pipefail
d=runs/r290-three-skills
prior="$d/stepping-directed-margin-formal"
model="$prior/model_200.pt"
pid=$(cat "$prior/job.pid")
until test -s "$model" && unzip -tq "$model" >/dev/null 2>&1; do
  kill -0 "$pid" || exit 1
  sleep 10
done
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
out="$d/stepping-attitude-stairs-formal"
mkdir "$out"
echo "$$" >"$out/job.pid"
exec env PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages" /isaac-sim/python.sh code/train_as2w_isaac_stepping.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs 256 --iterations 1200 --out "$out" --init-checkpoint "$model" --curriculum stairs-sequence-guarded --sequence-period 8 --sequence-stride .35 --sequence-shaped-reward --directed-stair-curriculum --packet-loss .02 --target-margin .05 --sequence-attitude --predictive-guard --seed 302 >"$out/job.log" 2>&1
