"""Compare independent reconstruction against native Genesis during real physics.

Normal reset and the frozen native actor remain the sole physical controllers.
The adapter reads state; it never writes robot poses or overrides native inputs.
"""
import sys,json,time,hashlib,types
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/root/gpufree-data/go2-supermimic/reproduction/animal_expression_v1')
sys.path.insert(0,str(ROOT));sys.path.insert(0,'/root/gpufree-data/skill_teachers/kine2go')
from rob2nav_kine_reference import KineReference,TeacherHistory
import genesis as gs
from motion_imitation.imitation_wrapper import ImitationWrapper
OUT=ROOT/'kine_observation_audit_v2';OUT.mkdir(exist_ok=True)
with (OUT/'started.json').open('x') as f:json.dump({'created_at':time.time()},f)
torch.set_num_threads(2);gs.init(seed=36501,backend=gs.gpu,logging_level='warning',performance_mode=False)
state={'phase':'running','student_training_started':False,'rows':[]}
def save():
 p=OUT/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'state.json')
save()
try:
 for clip in ('ai4_dog_trot_00','solo8_crawl_slow'):
  p=ROOT/'teachers'/clip;cfg=json.loads((p/'config.json').read_text());e=dict(cfg['env_cfg']);e['episode_length_s']=8.
  assert e.get('warmup_time_s',0)==0 and e.get('policy_steps_per_ref_motion_step',1)==1 and e['control_freq']==60
  lo,hi=cfg['policy_cfg']['motion_range'];env=ImitationWrapper([(str(p/'motion.npy'),lo,hi)],num_envs=4,env_cfg=e,obs_cfg=cfg['obs_cfg'],reward_cfg=cfg['reward_cfg'],command_cfg=cfg['command_cfg'],show_viewer=False,eval=True,debug=False,device='cuda')
  motion=torch.as_tensor(np.load(p/'motion.npy',allow_pickle=False)[lo:hi],device='cuda');ref=KineReference(motion)
  history=TeacherHistory(4,'cuda');clock=torch.zeros(4,device='cuda',dtype=torch.long);original_reset=env.reset_idx
  def reset(self,ids):
   original_reset(ids);history.reset(ids);clock[ids]=0
  env.reset_idx=types.MethodType(reset,env);env.reset()
  ck=torch.load(p/'logs/model.pt',map_location='cuda',weights_only=True)
  actor=torch.nn.Sequential(torch.nn.Linear(169,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,12)).cuda().eval()
  actor.load_state_dict({k.removeprefix('actor.'):v for k,v in ck['model_state_dict'].items() if k.startswith('actor.')})
  errors={'history':0.,'targets':0.,'actor_action':0.,'cycle_boundary_reference':0.};snapshots=[]
  with torch.inference_mode():
   for step in range(64):
    native=env.get_observations()['policy'];targets=ref.future_targets(clock,env.motion_start_offset_buf,env.motion_rot_offset_buf)
    reconstructed=history.teacher_observation(targets)
    errors['history']=max(errors['history'],float((native[:,:93]-reconstructed[:,:93]).abs().max()))
    errors['targets']=max(errors['targets'],float((native[:,93:]-targets).abs().max()))
    action=actor(native);errors['actor_action']=max(errors['actor_action'],float((actor(reconstructed)-action).abs().max()))
    snapshots.append({'native':native.cpu(),'reconstructed':reconstructed.cpu(),'clock':clock.cpu().clone(),'start_offset':env.motion_start_offset_buf.cpu().clone(),'heading_offset':env.motion_rot_offset_buf.cpu().clone()})
    clock=ref.advance(clock);env.step(action)
    assert torch.equal(clock,env.episode_length_buf),'Independent phase clock diverged'
    history.observe_after_physics(env.robot.get_pos(),env.robot.get_quat(),env.robot.get_qpos()[:,7:],env.actions)
   # Read-only boundary queries include both sides of a wrap and two cycles.
   ids=torch.arange(4,device='cuda');steps=torch.tensor([ref.length-1,ref.length,ref.length+1,2*ref.length+10],device='cuda')
   upstream=env._get_ref_state_at_frame(ids,steps%ref.length,steps//ref.length)
   native_ref=torch.cat((upstream['root_pos'],upstream['root_quat'],upstream['joint_rot']),-1)
   errors['cycle_boundary_reference']=float((native_ref-ref.at(steps,env.motion_rot_offset_buf)).abs().max())
  tensor_path=OUT/(clip+'_native_reconstruction.pt');torch.save(snapshots,tensor_path)
  row={'clip':clip,'num_envs':4,'control_steps':64,'physics_steps_per_control':4,'control_hz':60,'physical_controller':'frozen native actor on native observations','errors':errors,'clock_matches_native':True,'snapshots_sha256':hashlib.sha256(tensor_path.read_bytes()).hexdigest(),'observation_gate_passed':errors['history']<1e-6 and errors['targets']<1e-5 and errors['actor_action']<1e-4 and errors['cycle_boundary_reference']<1e-5}
  state['rows'].append(row);save();assert row['observation_gate_passed'],row
  del env
 state['phase']='completed_pending_review';state['limitations']=['Native Genesis reconstruction only, not MuJoCo transfer','No student policy or world-model reward/phase integration yet','No claim of blind student observation leakage validation yet']
except Exception as exc:state['phase']='failed_requires_review';state['error']=repr(exc)
state['updated_at']=time.time();save();print(json.dumps(state,indent=2),flush=True)
