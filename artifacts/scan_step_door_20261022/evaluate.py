import argparse,json,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--scenes',nargs='+',default=['stairs_door','stairs_route']);p.add_argument('--seeds',nargs='+',type=int,default=[200]);p.add_argument('--fail-fast',action='store_true');a=p.parse_args()
base='http://127.0.0.1:8769';out=Path('/root/gpufree-data/Rob2Nav/artifacts/scan_step_door_20261022');out.mkdir(parents=True,exist_ok=True)
def call(path,payload=None):
 req=urllib.request.Request(base+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json','X-Studio-Request':'1'})
 with urllib.request.urlopen(req,timeout=20) as r:return json.load(r)
results=[]
for scene in a.scenes:
 for seed in a.seeds:
  call('/api/control',dict(action='reset',scene=scene,task='navigate',seed=seed));end=time.monotonic()+30
  while time.monotonic()<end:
   s=call('/api/state')
   if s.get('scene')==scene and s['phase']=='READY' and s['seconds']<4:break
   time.sleep(.2)
  else:raise RuntimeError(s)
  call('/api/control',dict(action='start'));trace=[];end=time.monotonic()+220;printed=-1
  while time.monotonic()<end:
   s=call('/api/state');trace.append(s)
   stamp=int(s['seconds']//10)
   if stamp!=printed:
    print(scene,seed,'t',round(s['seconds'],1),'pos',[round(x,2) for x in s['position']],'d',round(s['distance'],2),'phase',s['phase'],'plans',(s.get('scan')or{}).get('plans'),flush=True);printed=stamp
   if s['phase'] in ('ARRIVED','FAULT','STOPPED'):break
   time.sleep(.2)
  call('/api/control',dict(action='stop'))
  passed=s['phase']=='ARRIVED' and not s['collision'] and not s['fallen'] and s['course_metrics']['steps_exited'] and (scene!='stairs_door' or s['course_metrics']['door_crossed'])
  passed=bool(passed and (s.get('scan')or{}).get('goal_height_enabled') and (s.get('scan')or{}).get('planner_max_velocity_mps')==.30)
  row=dict(scene=scene,seed=seed,passed=passed,phase=s['phase'],reason=s['reason'],time=s['seconds'],position=s['position'],distance=s['distance'],collision=s['collision'],fallen=s['fallen'],course=s['course_metrics'],routing_clearance_m=s.get('routing_clearance_m'),scan=s.get('scan'),hardware_enabled=False)
  (out/f'{scene}_{seed}_trace.json').write_text(json.dumps(trace))
  (out/f'{scene}_{seed}_summary.json').write_text(json.dumps(row,indent=2))
  results.append(row);print('RESULT',json.dumps(row),flush=True)
  if not passed and a.fail_fast:raise SystemExit('Full route failed; remaining cases not started')
(out/'summary.json').write_text(json.dumps(results,indent=2));print('DONE',flush=True)
