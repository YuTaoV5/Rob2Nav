import json,time,pathlib,urllib.request
p=pathlib.Path('outputs/backflip_triple_20261024');base='http://127.0.0.1:8765/api/live-skills/'
def call(path,payload=None):
 r=urllib.request.Request(base+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json','X-Studio-Request':'1'})
 return json.load(urllib.request.urlopen(r,timeout=20))
assert call('state').get('phase') not in ['STARTING','RUNNING']
steps=[{'skill':'stand','seconds':2}]+[{'skill':'backflip60','seconds':5,'trigger':True} for _ in range(3)]+[{'skill':'stand','seconds':2}]
launch=call('composition/start',{'steps':steps,'cycles':1,'compact_backflip_repeat':True});(p/'compact_web_e2e_launch.json').write_text(json.dumps(launch,indent=2));session=launch['id'];deadline=time.monotonic()+1200
while time.monotonic()<deadline:
 state=call('state');(p/'compact_web_e2e_state.json').write_text(json.dumps(state,indent=2))
 if state.get('id')!=session:raise RuntimeError('Session replaced externally; do not interfere')
 if state.get('phase') not in ['STARTING','RUNNING']:break
 time.sleep(2)
else:
 if call('state').get('id')==session:call('stop',{})
 raise RuntimeError('Own verification exceeded wall-time budget')
print(json.dumps({'phase':state.get('phase'),'frames':state.get('frames'),'result':state.get('result')}),flush=True)
