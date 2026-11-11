  trainer=train_as2w_isaac_stepping.py; count=256
  init="$d/stepping-go2-reset-exploration-init/model.pt"
  args=(--init-checkpoint "$init" --curriculum stairs-sequence-guarded --sequence-period 8 --sequence-stride .35 --sequence-shaped-reward --directed-stair-curriculum --packet-loss .02 --target-margin .05 --stance-margin-reward --sequence-attitude --predictive-guard --privileged-stair-critic --go2-support-reference "$d/go2-as2w-wheel-teacher-projected" --seed 305)
