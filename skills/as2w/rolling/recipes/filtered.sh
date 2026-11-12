# Sourced by training/as2w-backup/train.sh from the AS2-W workspace.
trainer=train_as2w_isaac_robust.py
count=24
default_iterations=20
init="$d/rolling-knee-margin-formal/model_1199.pt"
args=(--checkpoint "$init" --profile expanded --speed-cap .20 --speed-rolling
      --rolling-fixed-exit --rolling-stand-compensation --rolling-wheel-hold
      --park-precalibrated --park-continuous-bias --park-straight-bias
      --rolling-maneuvers --rolling-knee-margin --rolling-low-speed-smooth
      --rolling-leg-filter-hz 5 --pure-turn-common-scale .5 --safety
      --learning-rate .00005 --seed 311)
