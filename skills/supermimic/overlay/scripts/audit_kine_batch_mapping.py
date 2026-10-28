"""Frozen teacher batch parity on saved native inputs; no simulation or student."""
from pathlib import Path
import json,hashlib
import torch
ROOT=Path('/root/gpufree-data/go2-supermimic/reproduction/animal_expression_v1')
torch.set_float32_matmul_precision('highest');torch.set_num_threads(2)
rows=[]
for clip in ('ai4_dog_trot_00','solo8_crawl_slow'):
 p=ROOT/'teachers'/clip;c=json.loads((p/'config.json').read_text());e=c['env_cfg']
 ck=torch.load(p/'logs/model.pt',map_location='cpu',weights_only=True)
 net=torch.nn.Sequential(torch.nn.Linear(169,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,12)).eval()
 net.load_state_dict({k.removeprefix('actor.'):v for k,v in ck['model_state_dict'].items() if k.startswith('actor.')})
 inputs=[];expected=[];sources=[]
 for seed in range(36400,36404):
  d=ROOT/'native_independent'/clip/str(seed)
  report=json.loads((d/('native_screen_'+clip+'.json')).read_text())['trials'][0]
  trace_path=d/(clip+'_native_trace.json');trace=json.loads(trace_path.read_text())
  assert report['action_joint_names']==e['dof_names']
  sources.append({'seed':seed,'trace_sha256':hashlib.sha256(trace_path.read_bytes()).hexdigest(),'functional_candidate':report['functional_candidate']})
  for t in trace:
   if t['teacher_observation_before_step'] is not None:inputs.append(t['teacher_observation_before_step']);expected.append(t['action'])
 net=net.double()
 x=torch.tensor(inputs,dtype=torch.float64);y=torch.tensor(expected,dtype=torch.float64);assert x.shape==(32,169)
 with torch.inference_mode():
  serial=torch.cat([net(v[None]) for v in x]);batch=net(x)
 # An arbitrary SDK permutation forces name-based mapping and inverse mapping.
 target_names=sorted(e['dof_names']);idx=[e['dof_names'].index(n) for n in target_names]
 default=torch.tensor([e['default_joint_angles'][n] for n in e['dof_names']])
 mapped=(default+batch.clamp(-e['clip_actions'],e['clip_actions'])*e['action_scale'])[:,idx]
 inverse=[target_names.index(n) for n in e['dof_names']]
 recovered=(mapped[:,inverse]-default)/e['action_scale']
 parity=float((batch-y).abs().max());serial_error=float((batch-serial).abs().max());mapping_error=float((recovered-batch.clamp(-e['clip_actions'],e['clip_actions'])).abs().max())
 print('PARITY',clip,parity,serial_error,mapping_error,flush=True)
 assert parity<1e-4 and serial_error<1e-5 and mapping_error<1e-5
 rows.append({'clip':clip,'native_rows':len(inputs),'audit_inference_dtype':'float64 CPU; native saved actions float32 GPU','native_parity_tolerance':1e-4,'teacher_checkpoint_sha256':hashlib.sha256((p/'logs/model.pt').read_bytes()).hexdigest(),'source_inputs':sources,'batch_vs_saved_native_action_max_error':parity,'batch_vs_single_max_error':serial_error,'joint_target_roundtrip_max_error':mapping_error,'target_joint_names':target_names,'configured_action_latency':e['action_latency'],'executed_latency_control_steps':1 if e['action_latency']>0 else 0})
report={'phase':'completed','rows':rows,'simulation_executed':False,'observation_reconstruction_validated':False,'student_trained':False,'meaning':'saved native observation -> frozen actor batched inference and name-based target mapping only; not a MuJoCo closed-loop or phase-task validation'}
(ROOT/'kine_batch_mapping_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
