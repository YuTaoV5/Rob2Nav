  trainer=train_as2w_isaac_getup.py; count=384
  init="$d/getup-symmetric-formal/model_1199.pt"
  args=(--init-checkpoint "$init" --seed 301 --pose-bank "$d/supported-getup-poses.json" --side-pose-bank "$d/getup-training-mixed-side.json" --side-pose-probability .65 --staged-rise --rise-gain 85 --support-handoff --fixed-stand-integral --low-support-handoff --recovery-symmetry --packet-loss .02 --predictive-guard)
