set -euo pipefail
dest=runs/r286-robust
run="$dest/train-expanded-pilot"
p=$(cat "$run/job.pid")
while kill -0 "$p" 2>/dev/null; do sleep 3; done
grep -Fxq 'AS2W_ROBUST_COMPLETE expanded' "$run/job.log"
checkpoint="$run/model_299.pt"
test -s "$checkpoint"
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
for profile in nominal combined expanded; do
  out="$dest/eval-pilot-$profile"
  mkdir "$out"
  seed=286; envs=40
  if [ "$profile" = expanded ]; then seed=4286; envs=80; fi
  /isaac-sim/python.sh code/train_as2w_isaac_robust.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --out "$out" --checkpoint "$checkpoint" --profile "$profile" --num-envs "$envs" --iterations 1 --seed "$seed" --evaluate >"$out/job.log" 2>&1
  echo "DONE $out $(date -Is)"
done
echo R286_PILOT_COMPLETE
