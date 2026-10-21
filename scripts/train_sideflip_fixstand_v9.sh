#!/usr/bin/env bash
set -euo pipefail
cd "${ROB2NAV_ROOT:-/root/gpufree-data/Rob2Nav}"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
RUN="${1:-outputs/fixstand_sideflip_quiet_bridge_v9_20261022}"
BASE=(--backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip
  --unitree-source /root/gpufree-data/Rob2Nav-upstreams/unitree_rl_lab
  --motion outputs/fixstand_finetune_20261021/sideflip.npz
  --resume outputs/fixstand_sideflip_peak_pose_v8_20261022/model_7697.pt
  --headless --device cuda:0 --transfer-motors --fixstand-roundtrip --boundary-residual
  --boundary-residual-limit .20 --boundary-bridge-limit .05 --boundary-terminal-hip-taper-frames 50
  --boundary-blend-frames 10 --quiet-preparation --peak-endpoint-objective
  --finetune-learning-rate .00003 --reset-std .05)
/root/isaacsim/python.sh -m skills.sideflip.train "${BASE[@]}" --output "${RUN}_smoke" --num-envs 32 --iterations 2
test -f "${RUN}_smoke/completion.json"
/root/gpufree-data/envs/go2-mapless/bin/python - "${RUN}_smoke" <<'PY'
from pathlib import Path
import sys,json,yaml
from rob2nav.export_actor import ConfigLoader
p=Path(sys.argv[1]);env=yaml.load((p/'env.yaml').read_text(),Loader=ConfigLoader)
assert env['actions']['JointPositionAction']['boundary_blend_frames']==10
assert env['rewards']['quiet_preparation']['params']['end']==2.4
assert json.loads((p/'fixstand.json').read_text())['action_parameterization']=='reference_residual_v4'
print('VERIFIED_CONTINUOUS_BOUNDARY_AND_QUIET_REWARD',flush=True)
PY
/root/isaacsim/python.sh -m skills.sideflip.train "${BASE[@]}" --output "$RUN" --num-envs 2048 --iterations 1500
test -f "$RUN/completion.json"
/root/gpufree-data/envs/go2-mapless/bin/python - "$RUN" <<'PY'
from pathlib import Path
import sys,json,subprocess
from rob2nav.export_actor import export
p=Path(sys.argv[1]);c=max(p.glob('model_*.pt'),key=lambda x:int(x.stem.split('_')[-1]))
export(c,p/'exported');reports=[]
for seed in range(8900,8904):
    out=p/f'seed_{seed}'
    subprocess.run([sys.executable,'-m','rob2nav.mujoco_skill','--kind','sideflip',
        '--policy',str(p/'exported/policy.pt'),'--backflip-source','/root/gpufree-data/Rob2Nav-upstreams/GO2_backflip',
        '--motion','outputs/fixstand_finetune_20261021/sideflip.npz','--seconds','8','--physics-hz','1000',
        '--fixstand-roundtrip','--diagnostics','--seed',str(seed),'--output',str(out),*(['--video'] if seed==8900 else [])],check=True)
    d=json.loads((out/'summary.json').read_text());reports.append(dict(seed=seed,motion_pass=d['successes']==d['total'],
        preparation_pass=d['metrics']['preparation']['passed'],constraints_pass=d['hardware_constraints_pass'],
        passed=d['successes']==d['total'] and d['hardware_constraints_pass'],summary=str(out/'summary.json')))
    (p/'evaluation.json').write_text(json.dumps(dict(status='evaluating',episodes=reports,hardware_qualified=False),indent=2)+'\n')
(p/'evaluation.json').write_text(json.dumps(dict(status='passed_simulation_only' if all(x['passed'] for x in reports) else 'not_accepted',episodes=reports,hardware_qualified=False),indent=2)+'\n')
PY
