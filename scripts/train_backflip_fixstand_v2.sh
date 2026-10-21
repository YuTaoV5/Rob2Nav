#!/usr/bin/env bash
set -euo pipefail
cd "${ROB2NAV_ROOT:-/root/gpufree-data/Rob2Nav}"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
RUN="${1:-outputs/fixstand_backflip_entry_v2_20261021}"
if [[ -n "${WAIT_SIDEFLIP_PID:-}" ]]; then
  while ps -p "$WAIT_SIDEFLIP_PID" -o args= | grep -q 'skills.sideflip.train'; do sleep 20; done
fi
/root/isaacsim/python.sh -m skills.backflip.train_isaac51 \
  --source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip \
  --fixstand-roundtrip --delayed-entry --fixstand-output "$RUN" --resume \
  --checkpoint outputs/fixstand_finetune_20261021/backflip/model_5498.pt \
  --reset_optimizer --num_envs 1024 --max_iterations 1000 --trainer rsl_rl --headless --device cuda:0
test -f "$RUN/completion.json"
/root/gpufree-data/envs/go2-mapless/bin/python - "$RUN" <<'PY'
from pathlib import Path
import sys,json,subprocess
from skills.backflip.export_cpu import export
p=Path(sys.argv[1]);c=max(p.glob('model_*.pt'),key=lambda x:int(x.stem.split('_')[-1]))
export(c,p/'exported');reports=[]
for seed in range(9300,9304):
    out=p/f'seed_{seed}'
    subprocess.run([sys.executable,'-m','skills.backflip.simulate','--source','/root/gpufree-data/Rob2Nav-upstreams/GO2_backflip',
        '--onnx',str(p/'exported/policy.onnx'),'--seconds','5','--physics-hz','1000','--fixstand-roundtrip',
        '--seed',str(seed),'--output',str(out),*(['--video'] if seed==9300 else [])],check=True)
    d=json.loads((out/'summary.json').read_text());reports.append(dict(seed=seed,passed=d['successes']==d['total'],summary=str(out/'summary.json')))
    (p/'evaluation.json').write_text(json.dumps(dict(status='evaluating',episodes=reports,hardware_qualified=False),indent=2)+'\n')
(p/'evaluation.json').write_text(json.dumps(dict(status='passed_simulation_only' if all(x['passed'] for x in reports) else 'not_accepted',episodes=reports,hardware_qualified=False),indent=2)+'\n')
PY
