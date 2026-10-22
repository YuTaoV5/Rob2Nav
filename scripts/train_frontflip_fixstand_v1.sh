#!/usr/bin/env bash
set -euo pipefail
cd "${ROB2NAV_ROOT:-/root/gpufree-data/Rob2Nav}"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
RUN=outputs/fixstand_frontflip_positive_pitch_v1_20261022
BASE=(--source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip
 --flip-direction front --fixstand-roundtrip --delayed-entry --entry-hold-scale .1
 --urdf-speed-limits --fast-safety-curriculum --safety-start 0
 --initial-std .3 --min_action_std .1 --max_action_std .5
 --safety_warmup_iterations 100 --safety_ramp_iterations 4000 --seed 49
 --trainer rsl_rl --headless --device cuda:0)
/root/isaacsim/python.sh -m skills.backflip.train_isaac51 "${BASE[@]}" --fixstand-output "${RUN}_smoke" --num_envs 32 --max_iterations 2
test -f "${RUN}_smoke/completion.json"
/root/gpufree-data/envs/go2-mapless/bin/python - "${RUN}_smoke" <<'PY'
from pathlib import Path
import sys,yaml,json,hashlib
from rob2nav.export_actor import ConfigLoader
from skills.backflip.control import validate_entry_contract
p=Path(sys.argv[1]);d=yaml.load((p/'params/env.yaml').read_text(),Loader=ConfigLoader)
f=json.loads((p/'fixstand.json').read_text());validate_entry_contract(d,f)
assert d['flip_direction']==f['flip_direction']=='front' and f['skill']=='frontflip'
assert d['frontflip_reward_sha256']==hashlib.sha256(Path('skills/frontflip/task.py').read_bytes()).hexdigest()
assert yaml.safe_load((p/'params/agent.yaml').read_text())['policy']['init_noise_std']==.3
print('VERIFIED_INDEPENDENT_POSITIVE_PITCH_TASK',flush=True)
PY
/root/isaacsim/python.sh -m skills.backflip.train_isaac51 "${BASE[@]}" --fixstand-output "$RUN" --num_envs 4096 --max_iterations 5000
/root/gpufree-data/envs/go2-mapless/bin/python scripts/evaluate_fixstand_run.py --kind frontflip --run "$RUN" --seed-start 13600
