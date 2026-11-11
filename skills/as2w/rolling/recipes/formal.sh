  trainer=train_as2w_isaac_robust.py; count=384
  init="$d/rolling-maneuvers-formal/model_1199.pt"
  args=(--checkpoint "$init" --profile expanded --speed-cap 1 --speed-rolling --rolling-fixed-exit --rolling-stand-compensation --rolling-wheel-hold --park-precalibrated --park-continuous-bias --park-straight-bias --rolling-maneuvers --rolling-knee-margin --safety --learning-rate .00010 --seed 303)
