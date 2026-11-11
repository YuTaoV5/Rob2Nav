set -euo pipefail
d=runs/r290-three-skills
run="$d/getup-symmetric-formal"
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
dep="$PWD/../as2w-r284-release/deliveries/as2w-deploy"
bank="$d/supported-getup-holdout.json"
pid=$(cat "$run/job.pid")
for iter in 200 1199; do
  model="$run/model_$iter.pt"
  until test -s "$model" && unzip -tq "$model" >/dev/null 2>&1; do
    kill -0 "$pid" || exit 1
    sleep 10
  done
  a="$d/getup-symmetric-isaac-$iter"
  b="$d/getup-symmetric-mujoco-$iter"
  mkdir "$a" "$b"
  /isaac-sim/python.sh code/train_as2w_isaac_getup.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs 64 --iterations 1 --out "$a" --eval-checkpoint "$model" --seed 7297 --pose-bank "$bank" --staged-rise --rise-gain 85 --support-handoff --fixed-stand-integral --low-support-handoff --recovery-symmetry >"$a/job.log" 2>&1 &
  native_pid=$!
  PYTHONPATH="$PWD/code:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep" ../mujoco-venv/bin/python code/eval_as2w_mujoco_recovery.py assets/r237-train-root "$model" "$bank" "$b" --seed 7297 --low-support-handoff --paired-tilt >"$b/job.log" 2>&1 || echo "MuJoCo evaluation $iter failed"
  wait "$native_pid" || echo "Isaac Lab evaluation $iter failed"
  if test -s "$b/policy-observations.npy"; then
    ../mujoco-venv/bin/python code/export_as2w_candidate_onnx.py "$model" "$d/onnx-diagnostic/getup-symmetric-$iter" --clip 2 --observations "$b/policy-observations.npy" >"$b/onnx.log" 2>&1
  fi
done
