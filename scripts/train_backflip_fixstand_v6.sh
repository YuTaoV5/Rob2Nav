#!/usr/bin/env bash
set -euo pipefail
cd "${ROB2NAV_ROOT:-/root/gpufree-data/Rob2Nav}"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
RUN="${1:-outputs/fixstand_backflip_verified_entry_v6_20261022}"
BASE=(--source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip
  --fixstand-roundtrip --delayed-entry --entry-hold-scale .1
  --urdf-speed-limits --fast-safety-curriculum --safety-start 0
  --finetune-learning-rate .0001 --reset_curriculum --reset_optimizer
  --safety_warmup_iterations 100 --safety_ramp_iterations 1500
  --min_action_std .05 --max_action_std .20 --resume
  --checkpoint outputs/fixstand_backflip_quiet_entry_v3_20261022/model_7496.pt
  --trainer rsl_rl --headless --device cuda:0)
/root/isaacsim/python.sh -m skills.backflip.train_isaac51 "${BASE[@]}" --fixstand-output "${RUN}_smoke" --num_envs 32 --max_iterations 2
test -f "${RUN}_smoke/completion.json"
/root/gpufree-data/envs/go2-mapless/bin/python - "${RUN}_smoke" <<'PY'
from pathlib import Path
import sys,json,yaml
from rob2nav.export_actor import ConfigLoader
from skills.backflip.control import validate_entry_contract
p=Path(sys.argv[1]);env=yaml.load((p/'params/env.yaml').read_text(),Loader=ConfigLoader)
validate_entry_contract(env,json.loads((p/'fixstand.json').read_text()))
assert env['entry_hold_scale']==.1
print('VERIFIED_5S_EPISODE_1S_ENTRY_HOLD_0.1',flush=True)
PY
/root/isaacsim/python.sh -m skills.backflip.train_isaac51 "${BASE[@]}" --fixstand-output "$RUN" --num_envs 1024 --max_iterations 2000
test -f "$RUN/completion.json"
/root/gpufree-data/envs/go2-mapless/bin/python - "$RUN" <<'PY'
from pathlib import Path
import sys,json,subprocess
from skills.backflip.export_cpu import export
p=Path(sys.argv[1]);c=max(p.glob('model_*.pt'),key=lambda x:int(x.stem.split('_')[-1]))
export(c,p/'exported');reports=[]
for seed in range(10000,10004):
    out=p/f'seed_{seed}'
    subprocess.run([sys.executable,'-m','skills.backflip.simulate','--source','/root/gpufree-data/Rob2Nav-upstreams/GO2_backflip',
        '--onnx',str(p/'exported/policy.onnx'),'--seconds','5','--physics-hz','1000','--fixstand-roundtrip',
        '--seed',str(seed),'--output',str(out),*(['--video'] if seed==10000 else [])],check=True)
    d=json.loads((out/'summary.json').read_text());reports.append(dict(seed=seed,motion_pass=d['successes']==d['total'],
        constraints_pass=d['hardware_constraints_pass'],passed=d['successes']==d['total'] and d['hardware_constraints_pass'],summary=str(out/'summary.json')))
    (p/'evaluation.json').write_text(json.dumps(dict(status='evaluating',episodes=reports,hardware_qualified=False),indent=2)+'\n')
(p/'evaluation.json').write_text(json.dumps(dict(status='passed_simulation_only' if all(x['passed'] for x in reports) else 'not_accepted',episodes=reports,hardware_qualified=False),indent=2)+'\n')
PY
