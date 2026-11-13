#!/usr/bin/env bash
set -euo pipefail

root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
base="$root/training/as2w-backup/as2w-current"
snapshot="$base/runs/rolling/filtered-4000-env4096-stage020.ovvRL3/source"
recipe="$root/skills/as2w/rolling/recipes"
urdf=${AS2W_URDF:-"$base/assets/r237-train-root/official-model/robots/as2w_description/as2w.urdf"}
usd=${AS2W_USD:-"$base/runs/r280-isaac-flat-preflight/kit3-as2w.usd/as2w/as2w.usda"}
init="$root/deliveries/as2w-deploy/rolling-4000-stage020-research/weights/model_3999.pt"

case "${1:-}" in
  prepare)
    out=$(realpath -m "${2:?provide a new run directory}")
    test -s "$urdf" && test -s "$usd"
    printf '%s  %s\n' e46954b697ae8c4de2530ebae8f7c520f8f222bdd6a117195ff57bc4d3327769 "$init" | sha256sum -c -
    mkdir "$out"
    cp -a "$snapshot" "$out/source"
    python3 "$recipe/prepare_turn_20k.py" "$out/source"
    git -C "$root" rev-parse HEAD > "$out/git-revision.txt"
    printf '%s\n' "$init" > "$out/initial-checkpoint.txt"
    ( cd "$out"; find source -type f -print0 | sort -z | xargs -0 sha256sum ) > "$out/source.sha256"
    sha256sum "$usd" "$urdf" "$init" > "$out/assets.sha256"
    ;;
  train)
    out=$(realpath "${2:?provide the prepared run directory}")
    stage=${3:?provide smoke, 1, 2, 3, or 4}
    case "$stage" in
      smoke) cap=.20; updates=200; envs=${AS2W_SMOKE_ENVS:-512}; checkpoint=$init ;;
      1) cap=.20; updates=4000; envs=4096; checkpoint=$init ;;
      2) cap=.60; updates=6000; envs=4096; checkpoint="$out/stage-1/stage-final.pt" ;;
      3) cap=1.00; updates=6000; envs=4096; checkpoint="$out/stage-2/stage-final.pt" ;;
      4) cap=1.00; updates=4000; envs=4096; checkpoint="$out/stage-3/stage-final.pt" ;;
      *) echo "Unknown stage: $stage" >&2; exit 2 ;;
    esac
    ( cd "$out"; sha256sum --quiet -c source.sha256; sha256sum --quiet -c assets.sha256 )
    test -s "$checkpoint"
    mkdir "$out/stage-$stage"
    export PYTHONPATH="$out/source${PYTHONPATH:+:$PYTHONPATH}"
    cmd=("${AS2W_ISAAC_PY:?point to the tested Isaac Lab Python launcher}" "$out/source/train_as2w_isaac_robust.py"
         --viz none --device "${AS2W_DEVICE:-cuda:0}" --usd "$usd" --urdf "$urdf"
         --num-envs "$envs" --iterations "$updates" --out "$out/stage-$stage"
         --checkpoint "$checkpoint" --profile expanded --speed-cap "$cap"
         --speed-rolling --rolling-fixed-exit --rolling-stand-compensation --rolling-wheel-hold
         --park-precalibrated --park-continuous-bias --park-straight-bias --pure-turn-common-scale 0
         --rolling-leg-filter-hz 5 --rolling-low-speed-smooth --rolling-maneuvers
         --rolling-knee-margin --safety --learning-rate .00005 --seed 311)
    printf '%q ' "${cmd[@]}" > "$out/stage-$stage/launch.txt"
    printf '\n' >> "$out/stage-$stage/launch.txt"
    sha256sum "$checkpoint" > "$out/stage-$stage/initial.sha256"
    "${cmd[@]}" 2>&1 | tee "$out/stage-$stage/job.log"
    test -s "$out/stage-$stage/stage-final.pt"
    ;;
  diagnose)
    out=$(realpath "${2:?provide the prepared run directory}")
    checkpoint=$(realpath "${3:?provide a checkpoint}")
    report=$(realpath -m "${4:?provide a new diagnostics directory}")
    cap=${5:?provide the speed cap}
    ( cd "$out"; sha256sum --quiet -c source.sha256; sha256sum --quiet -c assets.sha256 )
    mkdir "$report"
    sha256sum "$checkpoint" > "$report/checkpoint.sha256"
    dep="$root/deliveries/as2w-deploy"
    export PYTHONPATH="$out/source:$dep/flat-r284-500-preview/runtime:$dep/omni-blind-stairs-r152/runtime:$dep${PYTHONPATH:+:$PYTHONPATH}"
    common=(--rolling-fixed-exit --rolling-stand-compensation --rolling-wheel-hold
            --park-precalibrated --park-continuous-bias --park-straight-bias
            --pure-turn-common-scale 0 --rolling-leg-filter-hz 5)
    "${AS2W_MUJOCO_PY:?point to the MuJoCo Python interpreter}" "$out/source/eval_as2w_mujoco_rolling.py" \
        "$base/assets/r237-train-root" "$checkpoint" "$report/mujoco" --speed "$cap" \
        "${common[@]}" --native-pd --substep-safety --acceptance --packet-loss .02 --trials 12
    "$AS2W_MUJOCO_PY" "$out/source/export_as2w_candidate_onnx.py" "$checkpoint" "$report/onnx" \
        --clip 6 --observations "$report/mujoco/policy-observations.npy"
    "${AS2W_ISAAC_PY:?point to the tested Isaac Lab Python launcher}" "$out/source/train_as2w_isaac_robust.py" \
        --viz none --device "${AS2W_DEVICE:-cuda:0}" --usd "$usd" --urdf "$urdf" \
        --num-envs 24 --iterations 1 --out "$report/isaac" --profile expanded --speed-cap "$cap" \
        --checkpoint "$checkpoint" --evaluate --seed 7393 --speed-rolling --safety \
        --rolling-low-speed-smooth "${common[@]}"
    ;;
  *) echo "Usage: $0 prepare RUN_DIR | train RUN_DIR {smoke|1|2|3|4} | diagnose RUN_DIR CHECKPOINT REPORT_DIR SPEED_CAP" >&2; exit 2 ;;
esac
