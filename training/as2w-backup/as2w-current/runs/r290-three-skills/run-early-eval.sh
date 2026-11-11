set -euo pipefail
d=runs/r290-three-skills
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
evaluate() {
  local name=$1 envs=$2 script=$3
  shift 3
  mkdir "$d/$name"
  echo "START $name $(date -Is)"
  /isaac-sim/python.sh "code/$script" --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs "$envs" --iterations 1 --out "$d/$name" "$@" >"$d/$name/job.log" 2>&1 || echo "FAILED $name"
  echo "DONE $name $(date -Is)"
}
evaluate rolling-eval-400 40 train_as2w_isaac_robust.py --checkpoint "$d/rolling-030-formal/model_400.pt" --profile expanded --speed-cap .30 --speed-rolling --safety --evaluate --seed 4290
evaluate stairs-eval-375 40 train_as2w_isaac_stepping.py --eval-checkpoint "$d/stairs-formal/model_375.pt" --seed 4291 --curriculum stairs
p=$(cat "$d/getup-formal/job.pid")
while [ ! -s "$d/getup-formal/model_100.pt" ]; do
  kill -0 "$p" 2>/dev/null || exit 2
  sleep 5
done
sleep 2
evaluate getup-eval-100 48 train_as2w_isaac_getup.py --eval-checkpoint "$d/getup-formal/model_100.pt" --seed 4292
echo R290_EARLY_EVAL_COMPLETE
