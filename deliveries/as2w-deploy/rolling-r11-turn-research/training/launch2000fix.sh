#!/usr/bin/env bash
# r11 turn-fix: 2000-iteration continuation from model_1999 using the FIXED
# training configuration (action-delay domain randomization disabled).
#
# Why this run exists
#   The previous turn-pilot run (PID 49391) used the same init and the same
#   command mix, but it was launched BEFORE the action-delay DR was fixed.  Its
#   DelayedTarget semantics aliased (apply_actions() runs once per physics
#   substep = 4x per 20 ms control step, while the depth-5 ring was designed for
#   one write per control step).  This run keeps everything identical except that
#   `as2w_action_delay_max_steps = 0`, which makes `delayed()` a pass-through.
#
#   Because both runs start at iteration 1999 and save every 100 iterations, the
#   checkpoint at model_2800 in each is at the same +801 update budget, so the
#   delay-on vs delay-off effect can be compared directly at equal training cost.
#
# Everything else is inherited unchanged: init checkpoint, command mix
# (hold .20 / straight .15 / spin .40 / arc .25), rewards, wheel drive (kp=0,
# kd=6, +-13.5 Nm), leg gains 55/2, 90% hard protection, save interval 100.
set -euo pipefail

root=/root/gpufree-data/Rob2Nav
base="$root/training/as2w-backup/as2w-current"
here=$(cd "$(dirname "$0")" && pwd)

init=/root/gpufree-data/as2w-runs/r11_fanziqi_lab/resume-20261115-204347-8722/model_1999.pt
expected_init=f54751faddc211b4cfa7498ac35e95e4f2617779bcdb8b3b5de5904c9da72985
printf '%s  %s\n' "$expected_init" "$init" | sha256sum -c -

# the patched environment file must be the fixed one
env_py="$base/code/as2w_isaac_robust_env.py"
expected_env=1af438db2ff3babc064a3a56bc8825880a5d3640b2d0f76174b7faa35233e24f
printf '%s  %s\n' "$expected_env" "$env_py" | sha256sum -c -

export AS2W_R11_SAVE_INTERVAL=100

envs=4096
iterations=2000
tag=train2000fix

out="$here/$tag-$(date +%Y%m%d-%H%M%S)-$$"
usd="$base/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"
urdf="$base/../dependencies/unitree_ros_as2w/robots/as2w_description/as2w.urdf"
pp=$(printf "%s:" /workspace/isaaclab/source/isaaclab*)
export PYTHONPATH="$base/code:${pp}/opt/isaaclab-venv/lib/python3.12/site-packages"
unset LD_LIBRARY_PATH
cd "$base"

cmd=(/isaac-sim/python.sh "$here/train_r11.py" --viz none --device cuda:0 --usd "$usd"
     --urdf "$urdf" --out "$out" --num-envs "$envs" --iterations "$iterations"
     --init-checkpoint "$init")

printf "%q " "${cmd[@]}"; printf "\n"

rc=0
"${cmd[@]}" || rc=$?

if [[ -d "$out" ]]; then
  git -C "$root" rev-parse HEAD > "$out/source-revision.txt" || true
  printf '%q ' "${cmd[@]}" > "$out/command.sh"; printf '\n' >> "$out/command.sh"
  {
    printf 'iterations=%s envs=%s save_interval=%s\n' "$iterations" "$envs" "${AS2W_R11_SAVE_INTERVAL}"
    printf 'init_checkpoint=%s\n' "$init"
    printf 'init_checkpoint_sha256=%s\n' "$expected_init"
    printf 'as2w_isaac_robust_env_sha256=%s\n' "$expected_env"
    printf 'action_delay_dr=DISABLED (as2w_action_delay_max_steps=0)\n'
    printf 'command_mix=hold:0.20 straight:0.15 spin:0.40 arc:0.25\n'
    printf 'prior_run_for_comparison=turn-pilot.xM3K4M (delay DR enabled, broken semantics)\n'
  } > "$out/run-params.txt"
fi
exit "$rc"
