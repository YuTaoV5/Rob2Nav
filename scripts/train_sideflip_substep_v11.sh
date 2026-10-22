#!/usr/bin/env bash
set -euo pipefail
cd "${ROB2NAV_ROOT:-/root/gpufree-data/Rob2Nav}"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
RUN=outputs/fixstand_sideflip_substep_peaks_v11_20261022
BASE=(--backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip
 --unitree-source /root/gpufree-data/Rob2Nav-upstreams/unitree_rl_lab
 --motion outputs/fixstand_finetune_20261021/sideflip.npz
 --resume outputs/fixstand_sideflip_landing_v10_20261022/model_10695.pt
 --headless --device cuda:0 --transfer-motors --fixstand-roundtrip --boundary-residual
 --boundary-residual-limit .20 --boundary-bridge-limit .05 --boundary-terminal-hip-taper-frames 50
 --boundary-blend-frames 10 --quiet-preparation --peak-endpoint-objective --landing-constraints
 --substep-objective --finetune-learning-rate .00001 --reset-std .03)
/root/isaacsim/python.sh -m skills.sideflip.train "${BASE[@]}" --output "${RUN}_smoke" --num-envs 32 --iterations 2
test -f "${RUN}_smoke/completion.json"
/root/gpufree-data/envs/go2-mapless/bin/python - "${RUN}_smoke" <<'PY'
from pathlib import Path
import sys,json,yaml
from rob2nav.export_actor import ConfigLoader
p=Path(sys.argv[1]);d=yaml.load((p/'env.yaml').read_text(),Loader=ConfigLoader)
assert json.loads((p/'fixstand.json').read_text())['physical_substep_objective'] is True
assert d['scene']['contact_forces']['update_period']==d['sim']['dt']==.002
s=json.loads((p/'physical_statistics.json').read_text())
assert s['physical_samples']>=s['control_steps']*10 and s['last_control_physical_samples']==10
print('VERIFIED_500HZ_PHYSICAL_PEAK_OBJECTIVE',flush=True)
PY
/root/isaacsim/python.sh -m skills.sideflip.train "${BASE[@]}" --output "$RUN" --num-envs 4096 --iterations 1800
/root/gpufree-data/envs/go2-mapless/bin/python scripts/evaluate_fixstand_run.py --kind sideflip --run "$RUN" --seed-start 13400
/root/gpufree-data/envs/go2-mapless/bin/python scripts/export_mirrored_sideflip.py --policy "$RUN/exported/policy.pt" \
 --motion outputs/fixstand_finetune_20261021/sideflip.npz --output "$RUN/right_reflected"
for seed in 13410 13411 13412 13413; do
 video=();if [[ "$seed" == 13410 ]]; then video=(--video);fi
 /root/gpufree-data/envs/go2-mapless/bin/python -m rob2nav.mujoco_skill --kind sideflip \
  --policy "$RUN/right_reflected/policy.pt" --motion "$RUN/right_reflected/rightflip.npz" \
  --backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip --seconds 8 --physics-hz 1000 \
  --fixstand-roundtrip --diagnostics --seed "$seed" "${video[@]}" --output "$RUN/right_seed_$seed"
done
/root/gpufree-data/envs/go2-mapless/bin/python - "$RUN" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);rows=[]
for s in range(13410,13414):
 d=json.loads((p/f'right_seed_{s}/summary.json').read_text());rows.append(dict(seed=s,passed=d['successes']==1 and d['hardware_constraints_pass']))
(p/'right_evaluation.json').write_text(json.dumps(dict(episodes=rows,passed=all(r['passed'] for r in rows),hardware_qualified=False),indent=2)+'\n')
PY
