"""Frozen right-flip checkpoint diagnostic; reserved final seeds remain unused."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch

root=Path.cwd()
sys.path.insert(0,str(root))
from rob2nav.export_actor import export
torch.set_num_threads(1)
run=root/'outputs/studio/runs/e4baa3665ca04e28'
out=root/'outputs/studio_diagnostics/right_v1_model200'
out.mkdir(parents=True,exist_ok=False)
export(run/'model_200.pt',out/'exported')
for name in ('env.yaml','agent.yaml','contract.json','fixstand.json'):
    shutil.copy2(run/name,out/name)
cmd=[sys.executable,'-m','rob2nav.mujoco_skill','--kind','sideflip',
    '--policy',str(out/'exported/policy.pt'),'--backflip-source',str(root.parent/'Rob2Nav-upstreams/GO2_backflip'),
    '--motion',str(root/'artifacts/web_continuity_iteration_20261023/inputs/right_v1/rightflip.npz'),
    '--seconds','8','--physics-hz','1000','--fixstand-roundtrip','--diagnostics',
    '--seed','27300','--output',str(out/'seed_27300')]
(out/'provenance.json').write_text(json.dumps(dict(scope='Mid-training diagnostic only',checkpoint=200,
    seed=27300,final_qualification=False,command=cmd),indent=2))
with (out/'execution.log').open('w') as log:
    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,
        env=dict(os.environ,MUJOCO_GL='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),check=True)
