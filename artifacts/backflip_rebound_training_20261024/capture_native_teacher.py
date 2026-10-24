"""Capture raw native actor observations/outputs without modifying control or physics.
Run with skills.backflip.simulate arguments; this does not create a qualified dataset.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import onnxruntime as ort
from skills.backflip.simulate import main

original = ort.InferenceSession
records=[]
class RecordingSession:
    def __init__(self,*args,**kwargs): self.inner=original(*args,**kwargs)
    def __getattr__(self,name): return getattr(self.inner,name)
    def run(self,*args,**kwargs):
        feed=kwargs.get('input_feed',args[1] if len(args)>1 else None)
        result=self.inner.run(*args,**kwargs)
        obs=np.asarray(next(iter(feed.values())))
        if obs.shape!=(1,60) or np.asarray(result[0]).shape!=(1,12):
            raise ValueError('Unexpected teacher actor layout')
        records.append((obs[0].copy(),np.asarray(result[0])[0].copy()))
        return result
ort.InferenceSession=RecordingSession
try:
    main()
finally:
    ort.InferenceSession=original
out=Path(sys.argv[sys.argv.index('--output')+1])
policy=Path(sys.argv[sys.argv.index('--onnx')+1])
if not records: raise RuntimeError('No teacher samples')
obs=np.stack([x[0] for x in records]);actions=np.stack([x[1] for x in records])
if not np.isfinite(obs).all() or not np.isfinite(actions).all():raise ValueError('Nonfinite teacher data')
np.savez_compressed(out/'teacher_samples.npz',observations=obs,raw_actions=actions,policy_time=np.arange(len(records))*.02)
summary=json.loads((out/'summary.json').read_text())
(out/'teacher_capture.json').write_text(json.dumps(dict(
    policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
    samples_sha256=hashlib.sha256((out/'teacher_samples.npz').read_bytes()).hexdigest(),
    capture_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    samples=len(records),observation_dim=60,action_dim=12,action_space='raw actor before entry mapping and clipping',
    physical_summary_success=summary['successes']==summary['total'],dataset_qualified=False,
    note='Single-seed teacher capture only; choose phase coverage and validate parity before training use.'),indent=2))
