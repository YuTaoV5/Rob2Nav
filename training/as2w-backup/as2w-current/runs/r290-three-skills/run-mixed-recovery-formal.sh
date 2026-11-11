set -euo pipefail
d=runs/r290-three-skills
prior="$d/getup-symmetric-formal"
model="$prior/model_1199.pt"
pid=$(cat "$prior/job.pid")
until test -s "$model" && unzip -tq "$model" >/dev/null 2>&1; do
  kill -0 "$pid" || exit 1
  sleep 10
done
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
out="$d/getup-mixed-side-formal"
mkdir "$out"
echo "$$" >"$out/job.pid"
exec env PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages" /isaac-sim/python.sh code/train_as2w_isaac_getup.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs 384 --iterations 1200 --out "$out" --init-checkpoint "$model" --seed 301 --pose-bank "$d/supported-getup-poses.json" --side-pose-bank "$d/getup-training-mixed-side.json" --side-pose-probability .65 --staged-rise --rise-gain 85 --support-handoff --fixed-stand-integral --low-support-handoff --recovery-symmetry --packet-loss .02 --predictive-guard >"$out/job.log" 2>&1
