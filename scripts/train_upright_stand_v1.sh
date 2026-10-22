#!/usr/bin/env bash
set -euo pipefail
cd "${ROB2NAV_ROOT:-/root/gpufree-data/Rob2Nav}"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
RUN=outputs/fixstand_upright_stand_static_v1_20261022
BASE=(--backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip
 --unitree-source /root/gpufree-data/Rob2Nav-upstreams/unitree_rl_lab
 --resume outputs/fixstand_upright_tight_endpoint_v3_20261022/model_2998.pt
 --headless --device cuda:0 --transfer-motors --stationary-upright --fixstand-roundtrip
 --endpoint-residual --endpoint-residual-limit .05 --upright-reset-fraction .5
 --reset-std .08 --finetune-learning-rate .00003)
/root/isaacsim/python.sh -m skills.upright_walk.train "${BASE[@]}" --output "${RUN}_smoke" --num-envs 32 --iterations 2
test -f "${RUN}_smoke/completion.json"
/root/gpufree-data/envs/go2-mapless/bin/python - "${RUN}_smoke" <<'PY'
from pathlib import Path
import sys,yaml,json
from rob2nav.export_actor import ConfigLoader
p=Path(sys.argv[1]);d=yaml.load((p/'env.yaml').read_text(),Loader=ConfigLoader)
assert all(tuple(d['commands']['base_velocity']['ranges'][key])==(0.,0.) for key in ['lin_vel_x','lin_vel_y','ang_vel_z'])
assert d['rewards']['stationary_balance']['weight']==12.
assert 'rear_alternation' not in d['rewards'] and 'rear_airtime' not in d['rewards']
assert json.loads((p/'fixstand.json').read_text())['stationary_upright'] is True
print('VERIFIED_STATIC_UPRIGHT_AND_ZERO_COMMAND',flush=True)
PY
/root/isaacsim/python.sh -m skills.upright_walk.train "${BASE[@]}" --output "$RUN" --num-envs 4096 --iterations 1800
/root/gpufree-data/envs/go2-mapless/bin/python scripts/evaluate_fixstand_run.py --kind upright_stand --run "$RUN" --seed-start 13100
