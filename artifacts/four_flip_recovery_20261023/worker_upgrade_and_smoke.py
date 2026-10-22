from pathlib import Path
import json,urllib.request,subprocess,time,os,signal
root=Path.cwd();out=root/'artifacts/four_flip_recovery_20261023'
from studio.jobs import Jobs
jobs=Jobs(root);old=135016
base='http://127.0.0.1:8765/api/'
def api(path,data=None):
 req=urllib.request.Request(base+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json','X-Studio-Request':'1'})
 with urllib.request.urlopen(req,timeout=30) as f:return json.load(f)
while True:
 active=[j for j in jobs.all() if j['state'] in ('running','preflight','evaluating','unknown')]
 if not active:break
 time.sleep(2)
proc=Path(f'/proc/{old}')
if proc.exists():
 cmd=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
 children=(proc/f'task/{old}/children').read_text().strip()
 if '-m studio.worker ' not in cmd or str(root) not in cmd or children:raise RuntimeError('Old worker identity/idle check failed; leave queue held')
 os.kill(old,signal.SIGTERM)
 deadline=time.monotonic()+10
 while proc.exists() and time.monotonic()<deadline:time.sleep(.1)
 # A zombie no longer owns a flock; the next worker still uses nonblocking flock.
record={}
name='four_flip_auto_eval_smoke_20261023'
existing=next((j for j in api('jobs') if j['name']==name),None)
if existing:smoke=existing
else:
 cfg=json.loads((root/'artifacts/step_airtime_recovery_20261023/front_v2_formal_config.json').read_text())
 cfg.update(name=name,num_envs=32,iterations=2,auto_evaluate=True)
 cfg['recipe_options']['evaluation_seed_start']=26300
 check=api('preflight',cfg)
 (out/'auto_eval_smoke_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
 (out/'auto_eval_smoke_preflight.json').write_text(json.dumps(check,ensure_ascii=False,indent=2))
 if not check['valid']:raise RuntimeError(str(check))
 smoke=api('jobs',cfg)
if smoke['state']=='draft':smoke=api('jobs/'+smoke['id']+'/submit',{})
record['smoke']={k:smoke[k] for k in ['id','state','name']}
(out/'worker_upgrade_resumed.json').write_text(json.dumps(record,indent=2))
for key in ['3e98ef1213334615','e4baa3665ca04e28']:
 j=jobs.get(key)
 if j['state']=='draft':j=api('jobs/'+key+'/submit',{})
 if j['state'] not in ('queued','preflight','running','succeeded'):raise RuntimeError('Unexpected held job state: '+str(j))
 record[key]=j['state']
 (out/'worker_upgrade_resumed.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
