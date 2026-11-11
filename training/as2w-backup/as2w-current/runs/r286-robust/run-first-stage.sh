set -euo pipefail
root=$PWD
dest=runs/r286-robust
seed_model=runs/r284-isaac-all-flat/model_1500.pt
usd="$root/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$root/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
launch() {
  local name=$1 checkpoint=$2 profile=$3 envs=$4 iters=$5
  shift 5
  local out="$dest/$name"
  mkdir "$out"
  echo "START $name $(date -Is)"
  /isaac-sim/python.sh code/train_as2w_isaac_robust.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --out "$out" --checkpoint "$checkpoint" --profile "$profile" --num-envs "$envs" --iterations "$iters" "$@" >"$out/job.log" 2>&1
  echo "DONE $name $(date -Is)"
}
for profile in nominal pose delay pose-delay friction combined; do
  launch "frozen-$profile" "$seed_model" "$profile" 40 1 --evaluate
done
for profile in baseline pose pose-delay; do
  launch "train-$profile" "$seed_model" "$profile" 512 300
  for scenario in nominal combined; do
    launch "eval-$profile-$scenario" "$dest/train-$profile/model_299.pt" "$scenario" 40 1 --evaluate
  done
done
echo R286_FIRST_STAGE_COMPLETE
