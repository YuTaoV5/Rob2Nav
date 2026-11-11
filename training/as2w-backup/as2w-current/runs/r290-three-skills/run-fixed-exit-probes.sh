set -euo pipefail
d=runs/r290-three-skills
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
dep="$PWD/../as2w-r284-release/deliveries/as2w-deploy"
native() {
  name=$1 script=$2 envs=$3
  shift 3
  out="$d/$name"
  mkdir "$out"
  env PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages" /isaac-sim/python.sh "code/$script" --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --num-envs "$envs" --iterations 1 --out "$out" "$@" >"$out/job.log" 2>&1
}
ckpt="$d/rolling-residual-formal/model_550.pt"
native rolling-fixed-exit-isaac-550 train_as2w_isaac_robust.py 40 --checkpoint "$ckpt" --profile expanded --speed-cap .30 --speed-rolling --rolling-fixed-exit --safety --evaluate --seed 5290 &
a=$!
native stairs-sequence-stance-probe train_as2w_isaac_stepping.py 16 --probe-dynamics --curriculum flat-sequence --seed 5293 &
b=$!
out="$d/rolling-fixed-exit-mujoco-550"
mkdir "$out"
env PYTHONPATH="$PWD/code:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep" ../mujoco-venv/bin/python code/eval_as2w_mujoco_rolling.py assets/r237-train-root "$ckpt" "$out" --speed .30 --rolling-fixed-exit >"$out/job.log" 2>&1 &
c=$!
rc=0
wait "$a" || rc=1
wait "$b" || rc=1
wait "$c" || rc=1
exit "$rc"
