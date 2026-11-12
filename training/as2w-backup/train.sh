#!/usr/bin/env bash
set -euo pipefail
base=$(cd "$(dirname "$0")" && pwd)
cd "$base/as2w-current"
[[ $# == 2 ]] || { echo "Usage: bash train.sh {rolling|rolling-filtered|stepping|recovery} NEW_OUTPUT_DIRECTORY" >&2; exit 2; }
d=runs/r290-three-skills
sim=${ISAAC_SIM_PYTHON:-/isaac-sim/python.sh}
pp=$(printf '%s:' "${ISAACLAB_SOURCE:-/workspace/isaaclab/source}"/isaaclab*)
export PYTHONPATH="${pp}${ISAACLAB_SITE_PACKAGES:-/opt/isaaclab-venv/lib/python3.12/site-packages}"
unset LD_LIBRARY_PATH
usd="$PWD/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$PWD/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
default_iterations=1200
case "$1" in
 rolling-filtered)
  recipe="$base/../../skills/as2w/rolling/recipes/filtered.sh"
  source "$recipe"
  ;;
 rolling|stepping|recovery)
  recipe="$base/../../skills/as2w/$1/recipes/formal.sh"
  source "$recipe"
  ;;
 *) echo "Unknown task: $1" >&2; exit 2 ;;
esac
for file in "$usd" "$urdf" "$init"; do
  [[ -s "$file" ]] || { echo "Missing input: $file" >&2; exit 1; }
done
[[ -x "$sim" ]] || { echo "Missing Isaac Sim interpreter: $sim" >&2; exit 1; }
mkdir -- "$2"
command=("$sim" "code/$trainer" --viz none --device "${AS2W_DEVICE:-cuda:0}" --usd "$usd" --urdf "$urdf" --num-envs "${AS2W_NUM_ENVS:-$count}" --iterations "${AS2W_ITERATIONS:-$default_iterations}" --out "$2" "${args[@]}")
git -C "$base" rev-parse HEAD > "$2/source-revision.txt" 2>/dev/null || printf 'unknown\n' > "$2/source-revision.txt"
sha256sum "$recipe" "$init" "$usd" "$urdf" > "$2/input-sha256.txt"
printf '%q ' "${command[@]}" > "$2/command.sh"
printf '\n' >> "$2/command.sh"
exec "${command[@]}"
