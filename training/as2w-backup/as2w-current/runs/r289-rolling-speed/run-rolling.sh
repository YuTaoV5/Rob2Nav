set -euo pipefail
dest=runs/r289-rolling-speed
cp=runs/r288-speed-range/train-030/model_299.pt
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
for pair in 030:.30 060:.60 100:1.00 150:1.50 200:2.00 250:2.50 300:3.00; do
  tag=${pair%:*}; speed=${pair#*:}
  train="$dest/train-$tag"; eval="$dest/eval-$tag"
  mkdir "$train" "$eval"
  echo "START $tag $(date -Is)"
  /isaac-sim/python.sh code/train_as2w_isaac_robust.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --out "$train" --checkpoint "$cp" --profile expanded --num-envs 512 --iterations 300 --seed 289 --learning-rate .00005 --speed-rolling --speed-cap "$speed" >"$train/job.log" 2>&1
  cp="$train/model_299.pt"
  /isaac-sim/python.sh code/train_as2w_isaac_robust.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --out "$eval" --checkpoint "$cp" --profile expanded --num-envs 40 --iterations 1 --evaluate --seed 4289 --speed-rolling --speed-cap "$speed" >"$eval/job.log" 2>&1
  echo "DONE $tag $(date -Is)"
  if ! python -c 'import json,sys;sys.exit(not json.load(open(sys.argv[1]))["gate"])' "$eval/speed-evaluation.json"; then
    echo "GATE_NOT_MET $tag"
    exit 0
  fi
done
echo R289_ROLLING_CURRICULUM_COMPLETE
