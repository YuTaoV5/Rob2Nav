#!/usr/bin/env bash
set -euo pipefail
cd "${ROB2NAV_ROOT:-/root/gpufree-data/Rob2Nav}"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
RUN=outputs/fixstand_upright_alternating_v5_20261022
BASE=(--backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip
 --unitree-source /root/gpufree-data/Rob2Nav-upstreams/unitree_rl_lab
 --resume outputs/fixstand_upright_tight_endpoint_v3_20261022/model_2998.pt
 --headless --device cuda:0 --transfer-motors --stepping-gait --alternating-rear-gait
 --fixstand-roundtrip --endpoint-residual --endpoint-residual-limit .05
 --upright-reset-fraction .5 --reset-std .15 --finetune-learning-rate .0001)
test -f outputs/fixstand_upright_tight_endpoint_v3_20261022/model_2998.pt
/root/isaacsim/python.sh -m skills.upright_walk.train "${BASE[@]}" --output "${RUN}_smoke" --num-envs 32 --iterations 2
test -f "${RUN}_smoke/completion.json"
/root/gpufree-data/envs/go2-mapless/bin/python - "${RUN}_smoke" <<'PY'
from pathlib import Path
import sys,yaml
from rob2nav.export_actor import ConfigLoader
p=Path(sys.argv[1]);d=yaml.load((p/'env.yaml').read_text(),Loader=ConfigLoader)
assert d['rewards']['rear_support']['weight']==0.
assert d['rewards']['rear_alternation']['weight']==6.
print('VERIFIED_ALTERNATING_REAR_OBJECTIVE',flush=True)
PY
/root/isaacsim/python.sh -m skills.upright_walk.train "${BASE[@]}" --output "$RUN" --num-envs 2048 --iterations 2000
/root/gpufree-data/envs/go2-mapless/bin/python scripts/evaluate_fixstand_run.py --kind upright_walk --run "$RUN" --seed-start 12200
