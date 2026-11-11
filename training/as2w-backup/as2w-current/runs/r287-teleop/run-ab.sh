set -euo pipefail
dest=runs/r287-teleop
start=runs/r286-robust/train-expanded-pilot/model_100.pt
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
job() {
  local name=$1 cp=$2 profile=$3 n=$4 it=$5
  shift 5
  local out="$dest/$name"
  mkdir "$out"
  echo "START $name $(date -Is)"
  /isaac-sim/python.sh code/train_as2w_isaac_robust.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --out "$out" --checkpoint "$cp" --profile "$profile" --num-envs "$n" --iterations "$it" "$@" >"$out/job.log" 2>&1
  echo "DONE $name $(date -Is)"
}
job frozen-nominal "$start" nominal 40 1 --evaluate --seed 4287
for arm in a b; do
  job train-$arm "$start" expanded 512 300 --teleop "$arm" --seed 287 --learning-rate .00005
  cp="$dest/train-$arm/model_299.pt"
  job eval-$arm-teleop "$cp" expanded 40 1 --evaluate --teleop-evaluate --seed 4287
  job eval-$arm-nominal "$cp" nominal 40 1 --evaluate --seed 4287
done
echo R287_AB_COMPLETE
