"""Check paired-pulse raw payloads and sign-invariant reset/prefix geometry."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
ROOT=Path(__file__).resolve().parents[1]/'reproduction/animal_expression_v1/animal_paired_excitation_v1';state=json.loads((ROOT/'state.json').read_text());assert state['phase']=='completed_pending_review';rows=[]
for fork in state['forks']:
 p=ROOT/f"pulse_prefix{fork['prefix']}_eps{fork['epsilon']:.2f}.pt";assert hashlib.sha256(p.read_bytes()).hexdigest()==fork['artifact_sha256'];d=torch.load(p,map_location='cpu',weights_only=False);q=d['pre_pulse_physical_qpos_relative_to_origin']['qpos'].double().reshape(26,4,19);quat=q[:,:,3:7];quat=quat/quat.norm(dim=-1,keepdim=True);dot=(quat[2:]*quat[:1]).sum(-1).abs().clamp(0,1);angle=2*torch.acos(dot);rawgap=(q[2:,:,3:7]-q[:1,:,3:7]).abs().max();t=d['tensors'];finite=all(bool(torch.isfinite(v).all()) for v in t.values() if v.is_floating_point());assert finite
 teacher_fields=[]
 def strings(x):
  if isinstance(x,str):teacher_fields.append(x)
  elif isinstance(x,dict):
   for v in x.values():strings(v)
  elif isinstance(x,(list,tuple)):
   for v in x:strings(v)
 strings(d);assert not any(any(word in s.lower() for word in ['password','bearer ','token=','private key']) for s in teacher_fields)
 rows.append({'prefix':fork['prefix'],'epsilon':fork['epsilon'],'all_tensors_finite':finite,'quaternion_component_max_gap':float(rawgap),'sign_invariant_orientation_max_angle_rad':float(angle.max()),'relative_root_position_max_gap':float((q[2:,:,:3]-q[:1,:,:3]).abs().max()),'joint_position_max_gap':float((q[2:,:,7:]-q[:1,:,7:]).abs().max()),'pulse_effect_to_baseline_repeat_velocity_noise_ratio':fork['real_two_step_effect_norm']*2*fork['epsilon']/max(fork['baseline_repeat_velocity_gap'],1e-12),'nonoverlap_8step_windows':104,'synthetic_robot_payload_only':True})
report={'rows':rows,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'policy_updated':False,'world_model_updated':False,'physics_executed':False,'limitations':'Quaternion q/-q raw components can differ near2 while orientation matches; sign-invariant geometry does not certify identical solver states. All mismatches/noise retained, no sample filtering. Synthetic simulation state/action payload only, not human telemetry or credentials.'};(ROOT/'geometry_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
