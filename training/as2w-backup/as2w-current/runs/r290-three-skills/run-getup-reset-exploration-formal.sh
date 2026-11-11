set -euo pipefail
d=runs/r290-three-skills
busy=$(cat "$d/rolling-knee-margin-formal/job.pid")
while kill -0 "$busy" 2>/dev/null; do sleep 15; done
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
out="$d/getup-reset-exploration-formal"
mkdir "$out"
echo "$$" >"$out/job.pid"
exec /isaac-sim/python.sh code/train_as2w_isaac_getup.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs 256 --iterations 1200 --out "$out" --init-checkpoint "$d/getup-reset-exploration-diagnostic/model_200.pt" --seed 306 --pose-bank "$d/supported-getup-poses.json" --side-pose-bank "$d/getup-training-mixed-side.json" --side-pose-probability .65 --staged-rise --rise-gain 85 --support-handoff --fixed-stand-integral --low-support-handoff --recovery-symmetry --packet-loss .02 --predictive-guard --settled-stand-reward >"$out/job.log" 2>&1
