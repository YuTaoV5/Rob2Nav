#!/usr/bin/env bash
set -euo pipefail
cd "${ROB2NAV_ROOT:-/root/gpufree-data/Rob2Nav}"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
CPU=/root/gpufree-data/envs/go2-mapless/bin/python
RUN="${1:-outputs/fixstand_upright_endpoint_v2_20261021}"
COMMON=(--backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip
  --unitree-source /root/gpufree-data/Rob2Nav-upstreams/unitree_rl_lab
  --resume artifacts/release_20261021/weights/upright_walk/representative_failed_baseline/model_4999.pt
  --headless --device cuda:0 --transfer-motors --stepping-gait --fixstand-roundtrip --endpoint-residual --reset-std .25)
/root/isaacsim/python.sh -m skills.upright_walk.train "${COMMON[@]}" --output "${RUN}_smoke" --num-envs 32 --iterations 2
test -f "${RUN}_smoke/completion.json"
/root/isaacsim/python.sh -m skills.upright_walk.train "${COMMON[@]}" --output "$RUN" --num-envs 1024 --iterations 1500
test -f "$RUN/completion.json"
"$CPU" - "$RUN" <<'PY'
import json,sys,subprocess
from pathlib import Path
from rob2nav.export_actor import export
p=Path(sys.argv[1]);c=max(p.glob('model_*.pt'),key=lambda x:int(x.stem.split('_')[-1]))
export(c,p/'exported')
reports=[]
for seed in range(9100,9104):
    out=p/f'seed_{seed}'
    subprocess.run([sys.executable,'-m','rob2nav.mujoco_skill','--kind','upright_walk',
        '--policy',str(p/'exported/policy.pt'),'--backflip-source','/root/gpufree-data/Rob2Nav-upstreams/GO2_backflip',
        '--seconds','8','--physics-hz','1000','--fixstand-roundtrip','--command','.15','0','0',
        '--seed',str(seed),'--output',str(out),*(['--video'] if seed==9100 else [])],check=True)
    d=json.loads((out/'summary.json').read_text());reports.append(dict(seed=seed,passed=d['successes']==d['total'],summary=str(out/'summary.json')))
    (p/'evaluation.json').write_text(json.dumps(dict(status='evaluating',episodes=reports,hardware_qualified=False),indent=2)+'\n')
(p/'evaluation.json').write_text(json.dumps(dict(status='passed_simulation_only' if all(x['passed'] for x in reports) else 'not_accepted',episodes=reports,hardware_qualified=False),indent=2)+'\n')
PY
