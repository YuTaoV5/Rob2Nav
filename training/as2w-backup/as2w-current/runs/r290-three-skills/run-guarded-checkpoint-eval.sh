set -euo pipefail
d=runs/r290-three-skills
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
dep="$PWD/../as2w-r284-release/deliveries/as2w-deploy"
wait_checkpoint() {
  local run=$1 iter=$2
  local model="$d/$run/model_$iter.pt" pid
  pid=$(cat "$d/$run/job.pid")
  until test -s "$model" && unzip -tq "$model" >/dev/null 2>&1; do
    kill -0 "$pid" || return 1
    sleep 10
  done
}
native() {
  local out=$1 script=$2 envs=$3
  shift 3
  mkdir "$out"
  /isaac-sim/python.sh "code/$script" --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs "$envs" --iterations 1 --out "$out" "$@" >"$out/job.log" 2>&1
}
rolling() {
  local iter=$1 run=rolling-fixed-exit-formal out ckpt
  wait_checkpoint "$run" "$iter"
  ckpt="$d/$run/model_$iter.pt"
  out="$d/rolling-fixed-exit-eval-$iter"
  native "$out" train_as2w_isaac_robust.py 40 --checkpoint "$ckpt" --profile expanded --speed-cap .30 --speed-rolling --rolling-fixed-exit --safety --evaluate --seed 6295
  PYTHONPATH="$PWD/code:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep" ../mujoco-venv/bin/python code/eval_as2w_mujoco_rolling.py assets/r237-train-root "$ckpt" "$out" --speed .30 --rolling-fixed-exit >"$out/mujoco.log" 2>&1
}
stepping() {
  local iter=$1 run=stepping-guarded-flat-formal
  wait_checkpoint "$run" "$iter"
  native "$d/stepping-guarded-flat-eval-$iter" train_as2w_isaac_stepping.py 40 --eval-checkpoint "$d/$run/model_$iter.pt" --curriculum flat-sequence-guarded --seed 6296
}
for iter in 200 1199; do
  rolling "$iter" & rolling_pid=$!
  stepping "$iter" & stepping_pid=$!
  wait "$rolling_pid" || echo "Rolling evaluation $iter failed"
  wait "$stepping_pid" || echo "Stepping evaluation $iter failed"
done
