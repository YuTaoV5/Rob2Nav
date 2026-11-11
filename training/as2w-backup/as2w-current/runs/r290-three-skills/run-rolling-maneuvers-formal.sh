set -euo pipefail
d=runs/r290-three-skills
smoke="$d/rolling-maneuvers-smoke"
prior="$d/rolling-fixed-exit-formal"
pid=$(cat "$prior/job.pid")
model="$prior/model_1199.pt"
until grep -q 'AS2W_ROBUST_COMPLETE' "$smoke/job.log"; do
  test ! -s "$smoke/job.pid" || kill -0 "$(cat "$smoke/job.pid")" || exit 1
  sleep 10
done
until test -s "$model" && unzip -tq "$model" >/dev/null 2>&1; do
  kill -0 "$pid" || exit 1
  sleep 10
done
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
out="$d/rolling-maneuvers-formal"
mkdir "$out"
echo "$$" >"$out/job.pid"
exec /isaac-sim/python.sh code/train_as2w_isaac_robust.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs 384 --iterations 1200 --out "$out" --checkpoint "$model" --profile expanded --speed-cap .60 --speed-rolling --rolling-fixed-exit --rolling-stand-compensation --rolling-wheel-hold --park-precalibrated --park-continuous-bias --park-straight-bias --rolling-maneuvers --safety --learning-rate .00010 --seed 303 >"$out/job.log" 2>&1
