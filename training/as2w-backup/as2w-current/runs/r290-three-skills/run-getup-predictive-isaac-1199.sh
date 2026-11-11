set -euo pipefail
d=runs/r290-three-skills
model="$d/getup-symmetric-formal/model_1199.pt"
unzip -tq "$model" >/dev/null
limit=$(cat /sys/fs/cgroup/memory.max)
if test "$limit" != max; then
  while test "$((limit-$(cat /sys/fs/cgroup/memory.current)))" -lt 11811160064; do sleep 20; done
fi
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
out="$d/getup-symmetric-predictive-isaac-1199"
mkdir "$out"
env PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages" /isaac-sim/python.sh code/train_as2w_isaac_getup.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs 64 --iterations 1 --out "$out" --eval-checkpoint "$model" --seed 7297 --pose-bank "$d/supported-getup-holdout.json" --staged-rise --rise-gain 85 --support-handoff --fixed-stand-integral --low-support-handoff --recovery-symmetry --packet-loss .02 --predictive-guard >"$out/job.log" 2>&1
b="$d/getup-symmetric-predictive-mujoco-1199"
onnx="$d/onnx-diagnostic/getup-symmetric-predictive-1199"
../mujoco-venv/bin/python code/export_as2w_candidate_onnx.py "$model" "$onnx" --clip 2 --observations "$b/policy-observations.npy" >"$b/onnx.log" 2>&1
../mujoco-venv/bin/python code/as2w_deploy_preflight.py "$d/deploy-reference/as2w-official-reference.json" "$onnx" --checkpoint "$model" --evaluation-contract "$out/contract.json" --evaluation-contract "$b/contract.json" --expect-evaluation predictive_guard=true --expect-evaluation observation_packet_loss=0.02
