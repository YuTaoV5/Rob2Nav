#!/usr/bin/env bash
set -euo pipefail
cd "${ROB2NAV_ROOT:-/root/gpufree-data/Rob2Nav}"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
RUN="${1:-outputs/fixstand_sideflip_v4_hip_taper_v7_20261022}"
# V5's endpoint and initial stance regressed; branch from the independently evaluated v4.
/root/isaacsim/python.sh -m skills.sideflip.train \
  --backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip \
  --unitree-source /root/gpufree-data/Rob2Nav-upstreams/unitree_rl_lab \
  --motion outputs/fixstand_finetune_20261021/sideflip.npz \
  --resume outputs/fixstand_sideflip_bridge_v4_20261021/model_6198.pt \
  --output "$RUN" --num-envs 1024 --iterations 1000 --headless --device cuda:0 \
  --transfer-motors --fixstand-roundtrip --boundary-residual \
  --boundary-residual-limit .20 --boundary-bridge-limit .05 \
  --boundary-terminal-hip-taper-frames 50
test -f "$RUN/completion.json"
/root/gpufree-data/envs/go2-mapless/bin/python - "$RUN" <<'PY'
from pathlib import Path
import sys,json,subprocess
from rob2nav.export_actor import export
p=Path(sys.argv[1]);c=max(p.glob('model_*.pt'),key=lambda x:int(x.stem.split('_')[-1]))
export(c,p/'exported');reports=[]
for seed in range(8700,8704):
    out=p/f'seed_{seed}'
    subprocess.run([sys.executable,'-m','rob2nav.mujoco_skill','--kind','sideflip',
        '--policy',str(p/'exported/policy.pt'),'--backflip-source','/root/gpufree-data/Rob2Nav-upstreams/GO2_backflip',
        '--motion','outputs/fixstand_finetune_20261021/sideflip.npz','--seconds','8','--physics-hz','1000',
        '--fixstand-roundtrip','--seed',str(seed),'--output',str(out),*(['--video'] if seed==8700 else [])],check=True)
    d=json.loads((out/'summary.json').read_text());reports.append(dict(seed=seed,motion_pass=d['successes']==d['total'],
        constraints_pass=d['hardware_constraints_pass'],passed=d['successes']==d['total'] and d['hardware_constraints_pass'],summary=str(out/'summary.json')))
    (p/'evaluation.json').write_text(json.dumps(dict(status='evaluating',episodes=reports,hardware_qualified=False),indent=2)+'\n')
(p/'evaluation.json').write_text(json.dumps(dict(status='passed_simulation_only' if all(x['passed'] for x in reports) else 'not_accepted',episodes=reports,hardware_qualified=False),indent=2)+'\n')
PY
