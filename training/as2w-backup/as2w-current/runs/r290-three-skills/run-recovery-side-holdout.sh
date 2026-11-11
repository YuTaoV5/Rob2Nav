set -euo pipefail
d=runs/r290-three-skills
run="$d/getup-symmetric-formal"
model="$run/model_1199.pt"
side="$d/supported-getup-side-holdout.json"
bank="$d/supported-getup-holdout.json"
pid=$(cat "$run/job.pid")
until test -s "$model" && unzip -tq "$model" >/dev/null 2>&1; do
  kill -0 "$pid" || exit 1
  sleep 10
done
test -s "$side"
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
dep="$PWD/../as2w-r284-release/deliveries/as2w-deploy"
a="$d/getup-symmetric-side-holdout-isaac-1199"
b="$d/getup-symmetric-side-holdout-mujoco-1199"
mkdir "$a" "$b"
PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages" /isaac-sim/python.sh code/train_as2w_isaac_getup.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs 64 --iterations 1 --out "$a" --eval-checkpoint "$model" --seed 9298 --pose-bank "$bank" --side-pose-bank "$side" --side-pose-probability 1 --staged-rise --rise-gain 85 --support-handoff --fixed-stand-integral --low-support-handoff --recovery-symmetry --packet-loss .02 >"$a/job.log" 2>&1 &
a_pid=$!
PYTHONPATH="$PWD/code:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep" ../mujoco-venv/bin/python code/eval_as2w_mujoco_recovery.py assets/r237-train-root "$model" "$bank" "$b" --seed 9298 --low-support-handoff --side-pose-bank "$side" --packet-loss .02 >"$b/job.log" 2>&1 &
b_pid=$!
rc=0
wait "$a_pid" || rc=1
wait "$b_pid" || rc=1
if test "$rc" -eq 0; then
  ../mujoco-venv/bin/python code/as2w_deploy_preflight.py "$d/deploy-reference/as2w-official-reference.json" "$d/onnx-diagnostic/getup-symmetric-1199" --checkpoint "$model" --evaluation-contract "$a/contract.json" --evaluation-contract "$b/contract.json"
fi
exit "$rc"
