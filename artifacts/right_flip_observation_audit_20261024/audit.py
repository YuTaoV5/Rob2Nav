import json,numpy as np,torch
from pathlib import Path
p=Path('outputs/studio/runs/fbf788e34b284900');s=torch.jit.load(str(p/'exported/policy.pt'),map_location='cpu').eval();torch.set_num_threads(1)
blocks={'reference_q':(0,12),'reference_dq':(12,24),'anchor_orientation':(24,30),'body_omega':(30,33),'joint_q':(33,45),'joint_dq':(45,57),'last_action':(57,69)}
def run(o):
 with torch.inference_mode():return s(torch.tensor(o,dtype=torch.float32)[None]).numpy()[0]
results=[]
for seed in range(20200,20204):
 t=json.loads((p/f'seed_{seed}/trace.json').read_text());i=int(np.argmax(np.max(abs(np.diff(np.array([x['target'] for x in t]),axis=0)),axis=1)))+1
 a=np.array(t[i-1]['observation']);b=np.array(t[i]['observation']);oa=run(a);ob=run(b);r={'seed':seed,'time':t[i]['time'],'replay_error':float(max(abs(ob-np.array(t[i]['action'])))),'raw_delta_RRhip':float(ob[3]-oa[3]),'blocks':{}}
 for k,(lo,hi) in blocks.items():
  c=b.copy();c[lo:hi]=a[lo:hi];out=run(c);r['blocks'][k]={'input_delta_max':float(max(abs(b[lo:hi]-a[lo:hi]))),'revert_delta_RRhip':float(ob[3]-out[3]),'remaining_jump_RRhip':float(out[3]-oa[3])}
 results.append(r)
print(json.dumps(results,indent=2))
