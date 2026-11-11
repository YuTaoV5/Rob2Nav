set -euo pipefail
dest=runs/r286-robust
pp=$(printf '%s:' /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
for i in 100 200; do
  checkpoint="$dest/train-expanded-pilot/model_$i.pt"
  for profile in nominal combined; do
    out="$dest/eval-pilot-$i-$profile"
    mkdir "$out"
    /isaac-sim/python.sh code/train_as2w_isaac_robust.py --viz none --device cuda:0 --usd "$usd" --urdf "$urdf" --out "$out" --checkpoint "$checkpoint" --profile "$profile" --num-envs 40 --iterations 1 --evaluate >"$out/job.log" 2>&1
    echo "DONE $out $(date -Is)"
  done
done
echo R286_INTERMEDIATE_COMPLETE
