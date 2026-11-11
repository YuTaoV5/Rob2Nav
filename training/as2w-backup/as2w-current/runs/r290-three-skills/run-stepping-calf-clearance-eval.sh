set -euo pipefail
d=runs/r290-three-skills
prefix=${1:-stepping-attitude}
run="$d/$prefix-formal"
if test -n "${2:-}"; then
  until test -s "$run/job.pid"; do kill -0 "$2" || exit 1; sleep 10; done
fi
pid=$(cat "$run/job.pid")
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
dep="$PWD/../as2w-r284-release/deliveries/as2w-deploy"
common=(--sequence-period 8 --sequence-stride .35 --packet-loss .02 --target-margin .05 --sequence-attitude --predictive-guard)
wait_room() {
  local limit
  limit=$(cat /sys/fs/cgroup/memory.max)
  if test "$limit" != max; then
    while test "$((limit-$(cat /sys/fs/cgroup/memory.current)))" -lt 9663676416; do sleep 20; done
  fi
}
iterations=(200 1199)
if test "${3:-}" = early-only; then iterations=(200); fi
for iter in "${iterations[@]}"; do
  model="$run/model_$iter.pt"
  until test -s "$model" && unzip -tq "$model" >/dev/null 2>&1; do
    kill -0 "$pid" || exit 1
    sleep 10
  done
  b="$d/$prefix-mujoco-holdout-$iter"
  mkdir "$b"
  trials=4
  if test "$iter" = 1199; then trials=8; fi
  PYTHONPATH="$PWD/code:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep" ../mujoco-venv/bin/python code/eval_as2w_mujoco_stepping.py assets/r237-train-root "$model" "$b" --seed 8302 --trials "$trials" --native-pd --leg-pd 55 2 --calf-mesh-diagnostic --down-approach .55 "${common[@]}" >"$b/job.log" 2>&1
  ../mujoco-venv/bin/python code/export_as2w_candidate_onnx.py "$model" "$d/onnx-diagnostic/$prefix-$iter" --clip 2 --observations "$b/policy-observations.npy" >"$b/onnx.log" 2>&1
  for mode in flat stairs; do
    wait_room
    a="$d/$prefix-isaac-$iter-$mode"
    mkdir "$a"
    extra=()
    if test "$mode" = stairs; then extra+=(--directed-stair-curriculum); fi
    /isaac-sim/python.sh code/train_as2w_isaac_stepping.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs 40 --iterations 1 --out "$a" --eval-checkpoint "$model" --eval-leg-pd 55 2 --curriculum "$mode-sequence-guarded" --sequence-shaped-reward --seed 6302 "${common[@]}" "${extra[@]}" >"$a/job.log" 2>&1
  done
  ../mujoco-venv/bin/python code/as2w_deploy_preflight.py "$d/deploy-reference/as2w-official-reference.json" "$d/onnx-diagnostic/$prefix-$iter" --checkpoint "$model" --evaluation-contract "$b/contract.json" --evaluation-contract "$d/$prefix-isaac-$iter-flat/contract.json" --evaluation-contract "$d/$prefix-isaac-$iter-stairs/contract.json" --expect-evaluation predictive_guard=true --expect-evaluation observation_packet_loss=0.02 --expect-evaluation sequence_attitude=true --expect-evaluation period_s=8 --expect-evaluation target_margin_rad=0.05
done
