"""Exercise explicit SCAN tasks; preserve failures and raw real-physics state."""
import argparse,json,time,urllib.request
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8765/scan-live-api');p.add_argument('--output',type=Path,required=True);p.add_argument('--scenes',default='');p.add_argument('--duration',type=float);p.add_argument('--seed',type=int,default=200);p.add_argument('--difficulty',type=int,choices=(1,2,3),default=2);p.add_argument('--person',choices=('person_01','person_02','person_03'),default='person_01');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 def request(path,data=None):
  q=urllib.request.Request(a.base+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json','X-Studio-Request':'1'})
  return json.load(urllib.request.urlopen(q,timeout=10))
 deadline=time.monotonic()+20
 while True:
  try:
   initial=request('/api/state')
   if initial.get('phase')!='INITIALIZING':break
  except OSError:
   if time.monotonic()>deadline:raise
  if time.monotonic()>deadline:raise RuntimeError('Worker initialization timeout')
  time.sleep(.3)
 results=[]
 for scene,mode,seconds in [('rooms_dynamic','navigate',65),('follow_obstacles','follow',22),('follow_stairs','follow',26),('follow_sand','follow',26),('follow_mixed','follow',32)]:
  if a.scenes and scene not in a.scenes.split(','):continue
  if a.duration is not None:seconds=min(120,max(1,a.duration))
  request('/api/control',dict(action='reset',scene=scene,task=mode,person_id=a.person,follow_distance=1.2,difficulty=a.difficulty,seed=a.seed))
  deadline=time.monotonic()+15
  while time.monotonic()<deadline:
   s=request('/api/state')
   if s.get('scene')==scene and s.get('phase')=='READY':break
   time.sleep(.1)
  assert s['scene']==scene and s['phase']=='READY',s
  request('/api/control',dict(action='start'));trace=[];start=time.monotonic()
  try:
   while time.monotonic()-start<seconds:
    s=request('/api/state');trace.append(s)
    if s['phase'] in ('ARRIVED','FAULT'):break
    time.sleep(.25)
  finally:request('/api/control',dict(action='stop'))
  positions=[s['position'] for s in trace];moved=max(((sum((p[i]-positions[0][i])**2 for i in range(2)))**.5 for p in positions),default=0)
  max_x=max((p[0] for p in positions),default=0)
  terrain_progress={}
  if scene in ('follow_stairs','follow_mixed'):terrain_progress['stairs']=dict(entered=max_x>=3.,exited=max_x>=6.75,exit_root_x_threshold_m=6.75)
  if scene in ('follow_sand','follow_mixed'):
   lo,hi=(2.7,6.4) if scene=='follow_sand' else (6.6,7.6)
   terrain_progress['sand_approximation']=dict(entered=max_x>=lo,exited=max_x>=hi+.35,exit_root_x_threshold_m=hi+.35)
  plans=max((s.get('scan',{}).get('plans',0) for s in trace if s.get('scan')),default=0)
  result=dict(scene=scene,task=mode,seed=a.seed,difficulty=a.difficulty,phase=s['phase'],distance=s['distance'],robot_displacement_m=moved,plans=plans,
   max_robot_x_m=max_x,max_robot_z_m=max((p[2] for p in positions),default=0),terrain_progress=terrain_progress,
   collision=any(t['collision'] for t in trace),fallen=any(t['fallen'] for t in trace),metrics=s.get('task_metrics'),
   selected_ids=sorted(set(t.get('selected_person','') for t in trace)),terrain=s.get('terrain'),samples=len(trace),
   reached_goal=s['phase']=='ARRIVED' if mode=='navigate' else None,scope='Fixed-seed bounded integration run; not robustness or hardware qualification')
  (a.output/(scene+'_trace.json')).write_text(json.dumps(trace));results.append(result);print(json.dumps(result),flush=True)
 (a.output/'summary.json').write_text(json.dumps(dict(episodes=results,hardware_enabled=False,person_source='simulated labelled tracks with line-of-sight',sand='rough rigid friction approximation only'),indent=2))
if __name__=='__main__':main()
