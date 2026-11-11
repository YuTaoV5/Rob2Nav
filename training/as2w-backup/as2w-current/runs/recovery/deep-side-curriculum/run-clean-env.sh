set -euo pipefail
d=runs/r290-three-skills
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
model="$d/getup-settled-stand-formal/model_1199.pt"
for stage in smoke formal; do
  envs=24
  budget=3
  if test "$stage" = formal; then envs=256; budget=1200; fi
  out="runs/recovery/deep-side-curriculum/$stage"
  if test "$stage" = smoke; then out="runs/recovery/deep-side-curriculum/smoke-retry"; fi
  mkdir "$out"
  echo "$$" >"$out/job.pid"
  /isaac-sim/python.sh code/train_as2w_isaac_getup.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs "$envs" --iterations "$budget" --out "$out" --init-checkpoint "$model" --seed 306 --pose-bank "$d/supported-getup-poses.json" --side-pose-bank "runs/recovery/deep-side-curriculum/training-poses.json" --side-pose-probability .65 --staged-rise --rise-gain 85 --support-handoff --fixed-stand-integral --low-support-handoff --recovery-symmetry --packet-loss .02 --predictive-guard --settled-stand-reward >"$out/job.log" 2>&1
done
