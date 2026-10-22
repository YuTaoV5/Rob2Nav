#!/usr/bin/env bash
set -euo pipefail
cd "${ROB2NAV_ROOT:-/root/gpufree-data/Rob2Nav}"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
RUN=outputs/fixstand_backflip_upstream_actor_v7_20261022
INIT=outputs/backflip_upstream_initialization_20261022
/root/gpufree-data/envs/go2-mapless/bin/python scripts/initialize_backflip_from_upstream.py \
 --source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip \
 --critic-parent outputs/fixstand_backflip_quiet_entry_v3_20261022/model_7496.pt --output "$INIT"
BASE=(--source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip
 --fixstand-roundtrip --delayed-entry --entry-hold-scale .1 --urdf-speed-limits
 --fast-safety-curriculum --safety-start 0 --finetune-learning-rate .00003
 --reset_curriculum --reset_optimizer --safety_warmup_iterations 100 --safety_ramp_iterations 1500
 --min_action_std .05 --max_action_std .15 --resume --checkpoint "$INIT/initialization.pt"
 --trainer rsl_rl --headless --device cuda:0)
/root/isaacsim/python.sh -m skills.backflip.train_isaac51 "${BASE[@]}" --fixstand-output "${RUN}_smoke" --num_envs 32 --max_iterations 2
test -f "${RUN}_smoke/completion.json"
/root/gpufree-data/envs/go2-mapless/bin/python - "${RUN}_smoke" <<'PY'
from pathlib import Path
import sys,json,yaml
from rob2nav.export_actor import ConfigLoader
from skills.backflip.control import validate_entry_contract
p=Path(sys.argv[1]);d=yaml.load((p/'params/env.yaml').read_text(),Loader=ConfigLoader)
validate_entry_contract(d,json.loads((p/'fixstand.json').read_text()))
assert d['entry_hold_scale']==.1
print('VERIFIED_UPSTREAM_ACTOR_AND_5S_ENTRY_CONTRACT',flush=True)
PY
/root/isaacsim/python.sh -m skills.backflip.train_isaac51 "${BASE[@]}" --fixstand-output "$RUN" --num_envs 1024 --max_iterations 2000
/root/gpufree-data/envs/go2-mapless/bin/python scripts/evaluate_fixstand_run.py --kind backflip --run "$RUN" --seed-start 12400
