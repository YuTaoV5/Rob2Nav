#!/usr/bin/env bash
set -euo pipefail
cd "${ROB2NAV_ROOT:-/root/gpufree-data/Rob2Nav}"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
RUN=outputs/fixstand_sideflip_landing_v10_20261022
PARENT=outputs/fixstand_sideflip_quiet_bridge_v9_20261022
checkpoint=$(find "$PARENT" -maxdepth 1 -name 'model_*.pt' | sort -V | tail -1)
test -n "$checkpoint"
BASE=(--backflip-source /root/gpufree-data/Rob2Nav-upstreams/GO2_backflip
 --unitree-source /root/gpufree-data/Rob2Nav-upstreams/unitree_rl_lab
 --motion outputs/fixstand_finetune_20261021/sideflip.npz --resume "$checkpoint"
 --headless --device cuda:0 --transfer-motors --fixstand-roundtrip --boundary-residual
 --boundary-residual-limit .20 --boundary-bridge-limit .05 --boundary-terminal-hip-taper-frames 50
 --boundary-blend-frames 10 --quiet-preparation --peak-endpoint-objective --landing-constraints
 --finetune-learning-rate .00001 --reset-std .05)
/root/isaacsim/python.sh -m skills.sideflip.train "${BASE[@]}" --output "${RUN}_smoke" --num-envs 32 --iterations 2
test -f "${RUN}_smoke/completion.json"
/root/gpufree-data/envs/go2-mapless/bin/python - "${RUN}_smoke" <<'PY'
from pathlib import Path
import sys,yaml
from rob2nav.export_actor import ConfigLoader
p=Path(sys.argv[1]);d=yaml.load((p/'env.yaml').read_text(),Loader=ConfigLoader)
assert d['rewards']['landing_constraints']['weight']==-8.
assert d['actions']['JointPositionAction']['boundary_blend_frames']==10
print('VERIFIED_QUIET_ENTRY_AND_LANDING_CONSTRAINTS',flush=True)
PY
/root/isaacsim/python.sh -m skills.sideflip.train "${BASE[@]}" --output "$RUN" --num-envs 1024 --iterations 1500
/root/gpufree-data/envs/go2-mapless/bin/python scripts/evaluate_fixstand_run.py --kind sideflip --run "$RUN" --seed-start 12300
