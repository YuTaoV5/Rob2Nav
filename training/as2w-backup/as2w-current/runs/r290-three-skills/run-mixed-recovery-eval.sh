set -euo pipefail
d=runs/r290-three-skills
prefix=${1:-getup-mixed}
run="$d/${2:-getup-mixed-side-formal}"
stage_pid=$(cat "$d/${3:-mixed-recovery-pipeline.pid}")
until test -s "$run/job.pid"; do kill -0 "$stage_pid" || exit 1; sleep 10; done
pid=$(cat "$run/job.pid")
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
dep="$PWD/../as2w-r284-release/deliveries/as2w-deploy"
bank="$d/supported-getup-holdout.json"
iterations=(200 1199)
if test "${4:-}" = early-only; then iterations=(200); fi
for iter in "${iterations[@]}"; do
  model="$run/model_$iter.pt"
  until test -s "$model" && unzip -tq "$model" >/dev/null 2>&1; do
    kill -0 "$pid" || exit 1
    sleep 10
  done
  cases=(random)
  if test "$iter" = 1199; then cases+=(shallow side); fi
  for group in "${cases[@]}"; do
    a="$d/$prefix-isaac-$iter-$group"
    b="$d/$prefix-mujoco-$iter-$group"
    mkdir "$a" "$b"
    native_args=()
    mujoco_args=()
    case "$group" in
      shallow) side="$d/supported-getup-shallow-holdout.json";;
      side) side="$d/supported-getup-side-holdout.json";;
      random) side=;;
    esac
    if test -n "$side"; then
      native_args=(--side-pose-bank "$side" --side-pose-probability 1.)
      mujoco_args=(--side-pose-bank "$side")
    else
      mujoco_args=(--paired-tilt)
    fi
    /isaac-sim/python.sh code/train_as2w_isaac_getup.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs 64 --iterations 1 --out "$a" --eval-checkpoint "$model" --seed 7301 --pose-bank "$bank" --staged-rise --rise-gain 85 --support-handoff --fixed-stand-integral --low-support-handoff --recovery-symmetry --packet-loss .02 --predictive-guard "${native_args[@]}" >"$a/job.log" 2>&1 &
    native_pid=$!
    PYTHONPATH="$PWD/code:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep" ../mujoco-venv/bin/python code/eval_as2w_mujoco_recovery.py assets/r237-train-root "$model" "$bank" "$b" --seed 7301 --low-support-handoff --packet-loss .02 --predictive-guard "${mujoco_args[@]}" >"$b/job.log" 2>&1 || echo "MuJoCo evaluation $iter/$group failed"
    wait "$native_pid" || echo "Isaac Lab evaluation $iter/$group failed"
    if test "$group" = random && test -s "$b/policy-observations.npy"; then
      ../mujoco-venv/bin/python code/export_as2w_candidate_onnx.py "$model" "$d/onnx-diagnostic/$prefix-$iter" --clip 2 --observations "$b/policy-observations.npy" >"$b/onnx.log" 2>&1
    fi
    if test -s "$a/getup-evaluation.json" && test -s "$b/mujoco-recovery-diagnostic.json"; then
      ../mujoco-venv/bin/python code/as2w_deploy_preflight.py "$d/deploy-reference/as2w-official-reference.json" "$d/onnx-diagnostic/$prefix-$iter" --checkpoint "$model" --evaluation-contract "$a/contract.json" --evaluation-contract "$b/contract.json" --expect-evaluation predictive_guard=true --expect-evaluation observation_packet_loss=0.02
    fi
  done
done
