set -euo pipefail
d=runs/r290-three-skills
name=rolling-knee-margin
run="$d/$name-formal"
model="$run/model_200.pt"
while ! test -s "$model" || ! unzip -tq "$model" >/dev/null 2>&1; do
  kill -0 "$(cat "$run/job.pid")" || exit 1
  sleep 15
done
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
dep="$PWD/../as2w-r284-release/deliveries/as2w-deploy"
common=(--rolling-fixed-exit --rolling-stand-compensation --rolling-wheel-hold --park-precalibrated --park-continuous-bias --park-straight-bias)
wait_room() {
  local limit
  limit=$(cat /sys/fs/cgroup/memory.max)
  if test "$limit" != max; then
    while test "$((limit-$(cat /sys/fs/cgroup/memory.current)))" -lt 9663676416; do sleep 20; done
  fi
}
evaluate() {
  local ckpt=$1 cap=$2 iter=$3 trials=$4
  local a="$d/$name-isaac-$iter-$cap" b="$d/$name-mujoco-$iter-$cap"
  local bundle="$d/onnx-diagnostic/$name-$iter-$cap"
  mkdir "$a" "$b"
  PYTHONPATH="$PWD/code:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep" ../mujoco-venv/bin/python code/eval_as2w_mujoco_rolling.py assets/r237-train-root "$ckpt" "$b" --speed "$cap" "${common[@]}" --native-pd --substep-safety --acceptance --packet-loss .02 --trials "$trials" >"$b/job.log" 2>&1
  wait_room
  env PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages" /isaac-sim/python.sh code/train_as2w_isaac_robust.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs "$((2*trials))" --iterations 1 --out "$a" --checkpoint "$ckpt" --profile expanded --speed-cap "$cap" --speed-rolling "${common[@]}" --safety --evaluate --seed 7303 >"$a/job.log" 2>&1
  ../mujoco-venv/bin/python code/export_as2w_candidate_onnx.py "$ckpt" "$bundle" --clip 6 --observations "$b/policy-observations.npy" >"$b/onnx.log" 2>&1
  ../mujoco-venv/bin/python code/as2w_deploy_preflight.py "$d/deploy-reference/as2w-official-reference.json" "$bundle" --checkpoint "$ckpt" --evaluation-contract "$b/mujoco-contract.json" --evaluation-contract "$a/contract.json" --expect-evaluation rolling_fixed_exit=true --expect-evaluation park_precalibrated=true --expect-evaluation park_straight_bias=true
}
evaluate "$model" 1 200 4
