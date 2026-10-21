#!/usr/bin/env bash
set -euo pipefail
cd "${ROB2NAV_ROOT:-/root/gpufree-data/Rob2Nav}"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
RUN="${1:-outputs/fixstand_upright_tight_endpoint_v3_20261022}"
/root/isaacsim/python.sh -m skills.upright_walk.train \
  --backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip \
  --unitree-source /root/gpufree-data/Rob2Nav-upstreams/unitree_rl_lab \
  --resume outputs/fixstand_upright_endpoint_v2_20261021/model_1499.pt \
  --output "$RUN" --num-envs 2048 --iterations 1500 --headless --device cuda:0 \
  --transfer-motors --stepping-gait --fixstand-roundtrip --endpoint-residual \
  --endpoint-residual-limit .05 --reset-std .20
test -f "$RUN/completion.json"
/root/gpufree-data/envs/go2-mapless/bin/python - "$RUN" <<'PY'
from pathlib import Path
import sys,json,subprocess
from rob2nav.export_actor import export
p=Path(sys.argv[1]);c=max(p.glob('model_*.pt'),key=lambda x:int(x.stem.split('_')[-1]))
export(c,p/'exported');reports=[]
for seed in range(9700,9704):
    out=p/f'seed_{seed}'
    subprocess.run([sys.executable,'-m','rob2nav.mujoco_skill','--kind','upright_walk',
        '--policy',str(p/'exported/policy.pt'),'--backflip-source','/root/gpufree-data/Rob2Nav-upstreams/GO2_backflip',
        '--seconds','8','--physics-hz','1000','--fixstand-roundtrip','--command','.15','0','0',
        '--seed',str(seed),'--output',str(out),*(['--video'] if seed==9700 else [])],check=True)
    d=json.loads((out/'summary.json').read_text());reports.append(dict(seed=seed,motion_pass=d['successes']==d['total'],
        constraints_pass=d['hardware_constraints_pass'],passed=d['successes']==d['total'] and d['hardware_constraints_pass'],summary=str(out/'summary.json')))
    (p/'evaluation.json').write_text(json.dumps(dict(status='evaluating',episodes=reports,hardware_qualified=False),indent=2)+'\n')
(p/'evaluation.json').write_text(json.dumps(dict(status='passed_simulation_only' if all(x['passed'] for x in reports) else 'not_accepted',episodes=reports,hardware_qualified=False),indent=2)+'\n')
PY
