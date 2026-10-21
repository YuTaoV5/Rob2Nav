import argparse,json,time,urllib.request,hashlib
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--base',default='http://127.0.0.1:8765/scan-live-api');parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
BASE=args.base.rstrip('/')
def req(path,body=None):
 return json.load(urllib.request.urlopen(urllib.request.Request(BASE+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json','X-Studio-Request':'1'}),timeout=8))
out=args.output;out.mkdir(parents=True,exist_ok=False)
results=[]
for scene,goal in [('empty',[2.,0.]),('slalom',[6.,1.6])]:
 req('/api/state');req('/api/control',{'action':'reset','scene':scene})
 deadline=time.monotonic()+10
 while time.monotonic()<deadline:
  s=req('/api/state')
  if s.get('scene')==scene and s.get('phase')=='READY' and s.get('seconds',999)<2:break
  time.sleep(.1)
 req('/api/control',{'action':'goal','goal':goal});time.sleep(.2)
 s=req('/api/state');assert s['goal']==goal and s['goal_valid'],s
 req('/api/control',{'action':'start'})
 trace=[];deadline=time.monotonic()+25
 while time.monotonic()<deadline:
  s=req('/api/state');trace.append(s)
  if s.get('phase') in ['ARRIVED','FAULT']:break
  time.sleep(.2)
 req('/api/control',{'action':'stop'})
 result=dict(scene=scene,requested_goal=goal,phase=s['phase'],distance=s['distance'],collision=s['collision'],fallen=s['fallen'],scan=s.get('scan'),samples=len(trace),
  success=s['phase']=='ARRIVED' and not s['collision'] and not s['fallen'],hardware_enabled=False)
 (out/(scene+'_trace.json')).write_text(json.dumps(trace))
 results.append(result);print(json.dumps(result),flush=True)
(out/'summary.json').write_text(json.dumps({'backend':'MuJoCo / RoboGauge MoE / native SCAN ROS2','scope':'Two fixed scenes and goals; integration checks, not independent-seed robustness','episodes':results,'successes':sum(x['success'] for x in results),'total':len(results),'hardware_enabled':False},indent=2))
