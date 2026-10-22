"""Queue one bounded damping continuation only after actual gain smoke passes."""
import json
from pathlib import Path
import subprocess
import time
import urllib.request

root=Path.cwd(); out=root/'artifacts/four_flip_recovery_20261023'
def api(path,data=None):
    req=urllib.request.Request('http://127.0.0.1:8765/api/'+path,
        data=None if data is None else json.dumps(data).encode(),
        headers={'Content-Type':'application/json','X-Studio-Request':'1'})
    return json.load(urllib.request.urlopen(req,timeout=30))

while True:
    jobs=api('jobs'); smoke=next(j for j in jobs if j['id']=='df310a187da04c32')
    if smoke['state'] in ('succeeded','failed','canceled','unknown'):break
    time.sleep(5)
if smoke['state']!='succeeded':raise RuntimeError('Gain smoke did not succeed; no formal continuation')
code="""import json,yaml,sys
from pathlib import Path
from rob2nav.export_actor import ConfigLoader
from skills.sideflip.gains import verify_export
r=Path('outputs/studio/runs/df310a187da04c32')
env=yaml.load((r/'env.yaml').read_text(),Loader=ConfigLoader);c=json.loads((r/'contract.json').read_text())
assert verify_export(env,c)==(40.,2.)
print(json.dumps(dict(actual_gain_contract_verified=True,kp=40,kd=2,capability_verified=False)))
"""
p=subprocess.run(['/root/gpufree-data/envs/go2-mapless/bin/python','-c',code],capture_output=True,text=True,check=True)
(out/'sideflip_gain_smoke_actual_audit.json').write_text(p.stdout)
cfg=json.loads((out/'sideflip_kd2_smoke_config.json').read_text())
cfg.update(name='sideflip_kd1p2_bounded_v13_20261023',num_envs=4096,iterations=200,auto_evaluate=True)
cfg['recipe_options'].update(joint_kd=1.2,evaluation_seed_start=27200)
check=api('preflight',cfg);(out/'sideflip_kd1p2_formal_preflight.json').write_text(json.dumps(check,indent=2))
if not check['valid']:raise RuntimeError(str(check))
(out/'sideflip_kd1p2_formal_config.json').write_text(json.dumps(cfg,indent=2))
job=next((j for j in api('jobs') if j['name']==cfg['name']),None)
if job is None:job=api('jobs',cfg)
if job['state']=='draft':job=api('jobs/'+job['id']+'/submit',{})
(out/'damping_training_pipeline_state.json').write_text(json.dumps(dict(formal=job['id'],state=job['state'],
    actual_smoke_audited=True,independent_seeds=[27200,27201,27202,27203],hardware_qualified=False),indent=2))
print(job['id'],job['state'],flush=True)
