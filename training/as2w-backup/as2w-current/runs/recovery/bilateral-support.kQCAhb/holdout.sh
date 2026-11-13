set -euo pipefail
out=$(dirname "$0")
pid=$(cat "$out/pid")
dep="$PWD/../as2w-r284-release/deliveries/as2w-deploy"
bank=runs/r290-three-skills/supported-getup-holdout.json
side=runs/r290-three-skills/supported-getup-side-holdout.json
for m in reward control; do
  until test -f "$out/$m.exit"; do
    kill -0 "$pid"
    sleep 5
  done
  test "$(cat "$out/$m.exit")" = 0
  model="$out/$m/model_24.pt"
  unzip -tq "$model" >/dev/null
  env -u LD_LIBRARY_PATH PYTHONPATH="$PWD/code:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep" ../mujoco-venv/bin/python code/eval_as2w_mujoco_recovery.py assets/r237-train-root "$model" "$bank" "$out/holdout-$m" --seed 7301 --family left --family right --side-pose-bank "$side" --low-support-handoff --packet-loss .02 --predictive-guard >"$out/holdout-$m.log" 2>&1
done
