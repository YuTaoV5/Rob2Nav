set -euo pipefail
d=runs/r290-three-skills
dep="$PWD/../as2w-r284-release/deliveries/as2w-deploy"
export PYTHONPATH="$PWD/code:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep"
for item in rolling-030-formal:400:6 stairs-formal:375:2 getup-formal:100:2; do
  IFS=: read -r run index clip <<< "$item"
  ../mujoco-venv/bin/python code/export_as2w_candidate_onnx.py "$d/$run/model_$index.pt" "$d/onnx-diagnostic/$run" --clip "$clip"
done
../mujoco-venv/bin/python code/eval_as2w_mujoco_rolling.py assets/r237-train-root "$d/rolling-030-formal/model_400.pt" "$d/rolling-eval-400" --speed .30
echo AS2W_CROSS_DIAGNOSTIC_COMPLETE
