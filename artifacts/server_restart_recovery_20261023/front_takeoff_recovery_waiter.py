from pathlib import Path
import json,urllib.request,subprocess,time,os,signal,shutil
from studio.jobs import Jobs
root=Path.cwd();out=root/'artifacts/four_flip_recovery_20261023';jobs=Jobs(root)
base='http://127.0.0.1:8765/api/'
def api(path,data=None):
 from studio.pipeline_api import request
 return request(path,data,base)
def submit(cfg,label):
 check=api('preflight',cfg);(out/(label+'_preflight.json')).write_text(json.dumps(check,ensure_ascii=False,indent=2))
 if not check['valid']:raise RuntimeError(str(check))
 (out/(label+'_config.json')).write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
 job=next((j for j in api('jobs') if j['name']==cfg['name']),None)
 if job is None:job=api('jobs',cfg)
 if job['state']=='draft':job=api('jobs/'+job['id']+'/submit',{})
 return job
smoke=jobs.get('0f602bd56467455c')
cfg=json.loads((out/'front_takeoff_smoke_config.json').read_text())
record=json.loads((out/'front_takeoff_pipeline_state.json').read_text())
while True:
 j=jobs.get(smoke['id'])
 if j['state'] in ('succeeded','failed','canceled','unknown'):break
 time.sleep(5)
if j['state']!='succeeded':raise RuntimeError('Reward smoke failed; do NOT queue formal task: '+str(j))
run=root/'outputs/studio/runs'/smoke['id']
code="""from pathlib import Path
import yaml,json
from rob2nav.export_actor import ConfigLoader
root=Path.cwd();run=Path(__import__('sys').argv[1]);baseline=root/'outputs/studio/runs/f01a531254144471/params/env.yaml'
a=yaml.load((run/'params/env.yaml').read_text(),Loader=ConfigLoader);b=yaml.load(baseline.read_text(),Loader=ConfigLoader)
changes={k:[b['rewards']['scales'][k],v] for k,v in a['rewards']['scales'].items() if b['rewards']['scales'][k]!=v}
assert changes=={'lin_vel_z':[20.,60.]},changes
assert a['episode_length_s']==5. and a['entry_delay_seconds']==1. and a['entry_hold_scale']==.1
assert a['flip_direction']=='front' and a['reward_scale_overrides']=={'lin_vel_z':60}
assert a['contact_sensor']['update_period']==a['sim']['dt']
assert a['contact_sensor']['history_length']>=a['decimation']
print(json.dumps(dict(passed=True,only_reward_changes=changes,episode_seconds=5,entry_seconds=1,entry_hold_scale=.1,flip_direction='front',capability_verified=False)))
"""
r=subprocess.run(['/root/gpufree-data/envs/go2-mapless/bin/python','-c',code,str(run)],cwd=root,capture_output=True,text=True,check=True)
(out/'front_takeoff_smoke_actual_audit.json').write_text(r.stdout)
shutil.copy2(run/'params/env.yaml',out/'front_takeoff_smoke_actual_env.yaml')
cfg.update(name='front_takeoff_weight_v3_stage1_20261023',num_envs=4096,iterations=600,auto_evaluate=True)
cfg['recipe_options']['evaluation_seed_start']=26400
formal=submit(cfg,'front_takeoff_formal');record['formal']=formal['id'];record['status']='queued_after_actual_smoke_passed'
(out/'front_takeoff_pipeline_state.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
