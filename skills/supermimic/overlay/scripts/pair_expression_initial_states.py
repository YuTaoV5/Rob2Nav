"""Canonical frozen pair; capture normal-reset state without setting any state.

Equal hashes certify only recorded fields. The study does not identify a causal
reward term or assert bitwise determinism across simulator launches.
"""
import sys,runpy,json,hashlib,time
from pathlib import Path
import torch
BASE=Path('/root/gpufree-data/go2-supermimic');sys.path.insert(0,str(BASE))
from mjlab.envs import ManagerBasedRlEnv
from go2_supermimic.supermimic import state as st
from go2_supermimic.supermimic.buffer import pack_generator
ROOT=BASE/'reproduction/animal_expression_v1';OUT=ROOT/'warmstart_pair_v2';OUT.mkdir(exist_ok=True)
original_reset=ManagerBasedRlEnv.reset;captures=[]
def digest(t):
 a=t.detach().cpu().contiguous().numpy()
 return {'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
def observed_reset(self,*a,**kw):
 result=original_reset(self,*a,**kw)
 record={'physics_dt':self.physics_dt,'control_dt':self.step_dt,'fields':{}}
 for name in ('qpos','qvel','ctrl','act','qacc','qacc_warmstart'):
  value=getattr(self.sim.data,name,None)
  if value is not None and hasattr(value,'detach'):record['fields'][name]=digest(value)
 assert all(n in record['fields'] for n in ('qpos','qvel','ctrl')),'Raw simulator state was not captured'
 record['fields']['actor_observation']=digest(result[0]['actor'])
 record['fields']['generator']=digest(pack_generator(self.command_manager.get_term('target').gen))
 c=st.robot_consts(self)
 for name in ('kp','kd','action_scale','default_q','effort'):
  record['fields'][name]=digest(getattr(c,name))
 captures.append(record);return result
ManagerBasedRlEnv.reset=observed_reset
models={'initialization':ROOT/'warmstart_actor_initialization_only.pt','smoke24':BASE/'logs/supermimic/2026-10-28_01-18-47_expression_rhythm_actorinit_smoke_v2_20261028/model_24.pt'}
state={'phase':'running','seed':36213,'num_envs':8,'training_started':False,'results':{}}
def save():
 p=OUT/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'state.json')
save()
try:
 for label,checkpoint in models.items():
  captures=[];output=OUT/(label+'.json')
  sys.argv=['eval_expression.py','--checkpoint',str(checkpoint),'--seed','36213','--num-envs','8','--output',str(output)]
  runpy.run_path(str(BASE/'scripts/eval_expression.py'),run_name='__main__')
  assert captures,'No reset captured'
  (OUT/(label+'_reset.json')).write_text(json.dumps({'reset_calls':captures},indent=2))
  state['results'][label]={'report':str(output),'normal_reset_calls':len(captures),'last_reset_snapshot':captures[-1]};save()
 a=state['results']['initialization']['last_reset_snapshot'];b=state['results']['smoke24']['last_reset_snapshot']
 state['differing_observed_fields']=[k for k in set(a['fields'])|set(b['fields']) if a['fields'].get(k)!=b['fields'].get(k)]
 state['observed_initial_fields_equal']=not state['differing_observed_fields'] and a['physics_dt']==b['physics_dt'] and a['control_dt']==b['control_dt']
 state['phase']='completed_pending_review';state['causal_reward_mechanism_proven']=False
except Exception as e:
 state['phase']='failed_requires_review';state['error']=repr(e)
finally:
 ManagerBasedRlEnv.reset=original_reset;state['updated_at']=time.time();save()
print(json.dumps(state,indent=2),flush=True)
