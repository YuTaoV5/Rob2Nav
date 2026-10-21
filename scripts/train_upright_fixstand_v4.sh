#!/usr/bin/env bash
set -euo pipefail
cd "${ROB2NAV_ROOT:-/root/gpufree-data/Rob2Nav}"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
PREVIOUS="${PREVIOUS_RUN:-outputs/fixstand_upright_tight_endpoint_v3_20261022}"
RUN="${1:-outputs/fixstand_upright_mixed_phase_v4_20261022}"
for ((minute=0; minute<480; minute++)); do
  if test -f "$PREVIOUS/evaluation.json"; then
    status=$(/root/gpufree-data/envs/go2-mapless/bin/python -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$PREVIOUS/evaluation.json")
    if [[ "$status" == "passed_simulation_only" ]]; then
      echo "V3_ACCEPTED; no v4 continuation needed"
      exit 0
    fi
    if [[ "$status" == "not_accepted" ]]; then
      break
    fi
  fi
  sleep 60
done
test "${status:-}" = "not_accepted" || { echo "V3_EVALUATION_TIMEOUT"; exit 1; }
checkpoint=$(/root/gpufree-data/envs/go2-mapless/bin/python -c 'from pathlib import Path;import sys;p=Path(sys.argv[1]);print(max(p.glob("model_*.pt"),key=lambda f:int(f.stem.split("_")[-1])))' "$PREVIOUS")
/root/isaacsim/python.sh -m skills.upright_walk.train \
  --backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip \
  --unitree-source /root/gpufree-data/Rob2Nav-upstreams/unitree_rl_lab \
  --resume "$checkpoint" --output "$RUN" --num-envs 2048 --iterations 1500 \
  --headless --device cuda:0 --transfer-motors --stepping-gait \
  --fixstand-roundtrip --endpoint-residual --endpoint-residual-limit .05 \
  --upright-reset-fraction .25 --reset-std .20
test -f "$RUN/completion.json"
/root/gpufree-data/envs/go2-mapless/bin/python - "$RUN" <<'PY'
from pathlib import Path
import sys,json,subprocess
from rob2nav.export_actor import export
p=Path(sys.argv[1]);c=max(p.glob('model_*.pt'),key=lambda x:int(x.stem.split('_')[-1]))
export(c,p/'exported');reports=[]
for seed in range(9800,9804):
    out=p/f'seed_{seed}'
    subprocess.run([sys.executable,'-m','rob2nav.mujoco_skill','--kind','upright_walk',
        '--policy',str(p/'exported/policy.pt'),'--backflip-source','/root/gpufree-data/Rob2Nav-upstreams/GO2_backflip',
        '--seconds','8','--physics-hz','1000','--fixstand-roundtrip','--command','.15','0','0',
        '--seed',str(seed),'--output',str(out),*(['--video'] if seed==9800 else [])],check=True)
    d=json.loads((out/'summary.json').read_text());reports.append(dict(seed=seed,motion_pass=d['successes']==d['total'],
        constraints_pass=d['hardware_constraints_pass'],passed=d['successes']==d['total'] and d['hardware_constraints_pass'],summary=str(out/'summary.json')))
    (p/'evaluation.json').write_text(json.dumps(dict(status='evaluating',episodes=reports,hardware_qualified=False),indent=2)+'\n')
(p/'evaluation.json').write_text(json.dumps(dict(status='passed_simulation_only' if all(x['passed'] for x in reports) else 'not_accepted',episodes=reports,hardware_qualified=False),indent=2)+'\n')
PY
