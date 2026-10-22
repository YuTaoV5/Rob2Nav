"""Inference-only diagnostic of an existing in-progress static balance run."""
from pathlib import Path
import hashlib,json,os,subprocess,sys

root=Path('/root/gpufree-data/Rob2Nav')
parent=root/'outputs/fixstand_upright_stand_static_v1_20261022'
checkpoint=parent/'model_4500.pt'
out=root/'outputs/upright_static_4500_diagnostic_20261022';out.mkdir(exist_ok=False)
from rob2nav.export_actor import export
export(checkpoint,out/'exported')
metadata=json.loads((out/'exported/policy.json').read_text())
assert metadata['fixstand_roundtrip']['stationary_upright']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(out/'provenance.json').write_text(json.dumps(dict(parent=str(parent),checkpoint=str(checkpoint),
 checkpoint_sha256=sha(checkpoint),policy_sha256=metadata['policy_sha256'],
 diagnostic_only=True,parent_training_in_progress=not (parent/'completion.json').exists(),
 selection='Second-most-recent checkpoint at initial process audit, before any evaluation',
 capability_verified=False,hardware_connected=False),indent=2))
rows=[]
for seed in (14800,14801):
 dest=out/f'seed_{seed}'
 with (out/f'seed_{seed}.log').open('w') as log:
  p=subprocess.run([sys.executable,'-m','rob2nav.mujoco_skill','--kind','upright_stand',
   '--policy',str(out/'exported/policy.pt'),'--backflip-source',str(root.parent/'Rob2Nav-upstreams/GO2_backflip'),
   '--command','0','0','0','--seconds','8','--physics-hz','1000','--fixstand-roundtrip','--diagnostics',
   '--seed',str(seed),'--output',str(dest)],cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=180,
   env=dict(os.environ,MUJOCO_GL='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
 row=dict(seed=seed,exit_code=p.returncode,passed=False)
 if p.returncode==0:
  path=dest/'summary.json';d=json.loads(path.read_text());d['diagnostic_only']=True
  path.write_text(json.dumps(d,indent=2));m=d['metrics']
  row.update(motion_pass=d['successes']==d['total'],constraints_pass=d['hardware_constraints_pass'],
    passed=d['successes']==d['total'] and d['hardware_constraints_pass'],standing=m.get('standing'),
    fixstand=m.get('fixstand'),speed_ratio=m.get('max_joint_speed_ratio'),
    limit_excess_rad=m.get('max_joint_limit_excess_rad'),body_contact=m.get('unexpected_body_ground_contact'))
 rows.append(row)
 (out/'comparison.json').write_text(json.dumps(dict(cases=rows,diagnostic_only=True,capability_verified=False),indent=2))
 print(json.dumps(row),flush=True)
(out/'completed.json').write_text(json.dumps(dict(diagnostic_cases=len(rows),new_training_submitted=False)))
