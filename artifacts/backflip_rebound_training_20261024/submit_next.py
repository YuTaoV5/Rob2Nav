from pathlib import Path
import json
from studio.pipeline_api import request
out=Path(__file__).resolve().parent
a=json.loads((out/'actual_smoke_audit.json').read_text())
assert a['b70bd35d97bc4e23']['scales']==dict(lin_vel_z=60.,ang_vel_x=-1.)
for name in ('cycle_smoke','front_bounded'):
 cfg=json.loads((out/(name+'_config.json')).read_text())
 check=request('preflight',cfg)
 (out/(name+'_preflight.json')).write_text(json.dumps(check,indent=2))
 if not check['valid']:raise RuntimeError(str(check))
 job=next((j for j in request('jobs') if j['name']==cfg['name']),None)
 if job is None:job=request('jobs',cfg)
 (out/(name+'_created.json')).write_text(json.dumps(job,indent=2))
 if job['state']=='draft':job=request('jobs/'+job['id']+'/submit',{})
 (out/(name+'_submission.json')).write_text(json.dumps(dict(id=job['id'],state=job['state']),indent=2))
 print(name,job['id'],job['state'],flush=True)
