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
wait_train rolling-residual-formal
evaluate rolling-residual-eval-1199 40 train_as2w_isaac_robust.py --checkpoint "$d/rolling-residual-formal/model_1199.pt" --profile expanded --speed-cap .30 --speed-rolling --rolling-residual --safety --evaluate --seed 6290
dep="$PWD/../as2w-r284-release/deliveries/as2w-deploy"
PYTHONPATH="$PWD/code:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep" ../mujoco-venv/bin/python code/eval_as2w_mujoco_rolling.py assets/r237-train-root "$d/rolling-residual-formal/model_1199.pt" "$d/rolling-residual-eval-1199" --speed .30 --rolling-residual >"$d/rolling-residual-eval-1199/mujoco.log" 2>&1
wait_train getup-supported-formal
test -s "$d/supported-getup-holdout.json"
evaluate getup-supported-eval-1199 64 train_as2w_isaac_getup.py --eval-checkpoint "$d/getup-supported-formal/model_1199.pt" --seed 6294 --pose-bank "$d/supported-getup-holdout.json" --staged-rise --rise-gain 85
echo R290_SECOND_FORMAL_EVAL_COMPLETE
