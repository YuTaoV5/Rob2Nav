set -euo pipefail
d=runs/r290-three-skills
run="$d/stepping-directed-stairs-formal"
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
dep="$PWD/../as2w-r284-release/deliveries/as2w-deploy"
pid=$(cat "$run/job.pid")
native() {
  local out=$1 terrain=$2 model=$3
  mkdir "$out"
  local extra=()
  if test "$terrain" = stairs; then extra=(--directed-stair-curriculum); fi
  /isaac-sim/python.sh code/train_as2w_isaac_stepping.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs 40 --iterations 1 --out "$out" --eval-checkpoint "$model" --curriculum "$terrain-sequence-guarded" --sequence-period 8 --sequence-stride .35 --sequence-shaped-reward --packet-loss .02 --seed 6299 "${extra[@]}" >"$out/job.log" 2>&1
}
for iter in 200 1199; do
  model="$run/model_$iter.pt"
  until test -s "$model" && unzip -tq "$model" >/dev/null 2>&1; do
    kill -0 "$pid" || exit 1
    sleep 10
  done
  native "$d/stepping-directed-stairs-flat-isaac-$iter" flat "$model" &
  flat_pid=$!
  native "$d/stepping-directed-stairs-stairs-isaac-$iter" stairs "$model" &
  stairs_pid=$!
  out="$d/stepping-directed-stairs-mujoco-$iter"
  mkdir "$out"
  trials=4
  if test "$iter" = 1199; then trials=8; fi
  PYTHONPATH="$PWD/code:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep" ../mujoco-venv/bin/python code/eval_as2w_mujoco_stepping.py assets/r237-train-root "$model" "$out" --seed 8298 --trials "$trials" --sequence-period 8 --sequence-stride .35 --packet-loss .02 >"$out/job.log" 2>&1 || echo "MuJoCo evaluation $iter failed"
  wait "$flat_pid" || echo "Isaac flat evaluation $iter failed"
  wait "$stairs_pid" || echo "Isaac stairs evaluation $iter failed"
  if test -s "$out/policy-observations.npy"; then
    ../mujoco-venv/bin/python code/export_as2w_candidate_onnx.py "$model" "$d/onnx-diagnostic/stepping-directed-$iter" --clip 2 --observations "$out/policy-observations.npy" >"$out/onnx.log" 2>&1
  fi
done
