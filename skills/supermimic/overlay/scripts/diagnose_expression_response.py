"""Run canonical evaluations, then read their collected arrays after physics ends.

Steady statistics exclude the first two seconds only in this diagnostic. The raw
functional/physical reports still contain every sample. No new training occurs.
"""
import sys,runpy,json
from pathlib import Path
import numpy as np
BASE=Path('/root/gpufree-data/go2-supermimic');ROOT=BASE/'reproduction/animal_expression_v1'
state=json.loads((ROOT/'expression_formal/state.json').read_text());checkpoint=state['checkpoint']
rows=[]
for role in ('teacher','student'):
 sys.argv=['eval_expression.py','--seed','36130','--num-envs','8','--output',str(ROOT/(role+'_response_raw.json'))]
 if role=='student':sys.argv+=['--checkpoint',checkpoint]
 data=runpy.run_path(str(BASE/'scripts/eval_expression.py'),run_name='__main__')
 vel=data['vel'].numpy();cmd=data['cmd'].numpy();n=vel.shape[1];begin=100
 for style,label in enumerate(data['task'].STYLES):
  mask=np.arange(n)%4==style;x=cmd[begin:,mask];y=vel[begin:,mask]
  metrics=[]
  for axis in range(3):
   xx=x[...,axis].reshape(-1);yy=y[...,axis].reshape(-1);var=np.var(xx)
   metrics.append({'axis':('vx','vy','wz')[axis],'command_mean':float(xx.mean()),'actual_mean':float(yy.mean()),'command_std':float(xx.std()),'actual_std':float(yy.std()),'response_gain':float(np.mean((xx-xx.mean())*(yy-yy.mean()))/var) if var>1e-8 else None,'correlation':float(np.corrcoef(xx,yy)[0,1]) if var>1e-8 and np.var(yy)>1e-8 else None})
  rows.append({'role':role,'style':label,'steady_from_seconds':2.,'metrics':metrics})
 out={'seed':36130,'num_envs':8,'diagnostic_only':True,'raw_reports_keep_all_samples':True,'student_checkpoint':checkpoint,'rows':rows,'causal_reward_mechanism_proven':False}
 (ROOT/'response_diagnostic.json').write_text(json.dumps(out,indent=2))
 print(role,'response statistics saved',flush=True)
