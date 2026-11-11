root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
old="$root/isaaclab-2.3.2/source"
export LD_LIBRARY_PATH="/isaac-sim/kit/python/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$old/isaaclab:$old/isaaclab_assets:$old/isaaclab_tasks:$old/isaaclab_rl:$old/isaaclab_mimic:$root/legacy-python:/opt/isaaclab-venv/lib/python3.12/site-packages"
