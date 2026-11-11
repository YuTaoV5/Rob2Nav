set -eu
d=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export LD_LIBRARY_PATH="/isaac-sim/kit/python/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$(find /workspace/isaaclab/source -mindepth 1 -maxdepth 1 -type d -name 'isaaclab*' | paste -sd:):/opt/isaaclab-venv/lib/python3.12/site-packages"
urdf=$(realpath assets/official-model/robots/as2w_description/as2w.urdf)
exec /isaac-sim/python.sh code/train_as2w_isaac_flat_all.py --device cuda:0 --usd "$d/kit3-as2w.usd/as2w/as2w.usda" --urdf "$urdf" "$@"
