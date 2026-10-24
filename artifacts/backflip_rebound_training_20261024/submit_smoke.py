from pathlib import Path
import hashlib,json
from studio.pipeline_api import request
out=Path(__file__).resolve().parent
cfg=json.loads((out/'config.json').read_text())
root=out.parents[1]
assert hashlib.sha256((root/cfg['checkpoint']).read_bytes()).hexdigest()=='bf218a18e0fc6aa44534f7b32a20aae742ad9da2795af1bab034147451ed0bb1'
assert 'rebound_cycles' in json.dumps(request('schema'))
check=request('preflight',cfg)
(out/'preflight.json').write_text(json.dumps(check,indent=2))
if not check['valid']:raise RuntimeError(str(check))
job=next((j for j in request('jobs') if j['name']==cfg['name']),None)
if job is None:job=request('jobs',cfg)
(out/'created.json').write_text(json.dumps(job,indent=2))
if job['state']=='draft':job=request('jobs/'+job['id']+'/submit',{})
(out/'submission.json').write_text(json.dumps(dict(id=job['id'],state=job['state'],scope='configuration/runtime smoke only; no capability acceptance'),indent=2))
print(job['id'],job['state'])
