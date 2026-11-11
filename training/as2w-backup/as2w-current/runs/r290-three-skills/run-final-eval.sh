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
wait_train() {
  local run=$1 p
  p=$(cat "$d/$run/job.pid")
  while kill -0 "$p" 2>/dev/null; do sleep 5; done
  test -s "$d/$run/model_1199.pt"
}
wait_train stairs-formal
evaluate stairs-eval-1199 40 train_as2w_isaac_stepping.py --eval-checkpoint "$d/stairs-formal/model_1199.pt" --seed 5291 --curriculum stairs
wait_train rolling-030-formal
evaluate rolling-eval-1199 40 train_as2w_isaac_robust.py --checkpoint "$d/rolling-030-formal/model_1199.pt" --profile expanded --speed-cap .30 --speed-rolling --safety --evaluate --seed 5290
wait_train getup-formal
evaluate getup-eval-1199 48 train_as2w_isaac_getup.py --eval-checkpoint "$d/getup-formal/model_1199.pt" --seed 5292
echo R290_FIRST_FORMAL_EVAL_COMPLETE
