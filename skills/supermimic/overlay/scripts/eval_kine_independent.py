"""Native Genesis closed-loop screening, frozen public weights, JSON configuration.

No trajectory playback / state writes after normal reset. Functional and physical
metrics are separate. This program does not train a SuperMimic student.
"""
import sys, json, hashlib, time, os
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/root/gpufree-data/skill_teachers/kine2go')
import genesis as gs
from motion_imitation.imitation_wrapper import ImitationWrapper
ROOT=Path('/root/gpufree-data/go2-supermimic/reproduction/animal_expression_v1')
OUTPUT=Path(os.environ['KINE_SCREEN_OUTPUT']);OUTPUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
gs.init(seed=int(os.environ['KINE_SCREEN_SEED']),backend=gs.gpu,logging_level='warning',performance_mode=False)
rows=[]
for clip in sys.argv[1:] or ['ai4_dog_trot_00']:
 p=ROOT/'teachers'/clip;c=json.loads((p/'config.json').read_text());e=c['env_cfg'].copy();e['episode_length_s']=8.
 start,end=c['policy_cfg']['motion_range'];motion=np.load(p/'motion.npy',allow_pickle=False)
 lo,hi,_=slice(start,end).indices(len(motion))
 assert 0<=lo<hi<=len(motion),(clip,start,end,motion.shape)
 env=ImitationWrapper([(str(p/'motion.npy'),start,end)],num_envs=1,env_cfg=e,obs_cfg=c['obs_cfg'],reward_cfg=c['reward_cfg'],command_cfg=c['command_cfg'],show_viewer=False,eval=True,debug=False,device='cuda')
 ckpt=torch.load(p/'logs/model.pt',map_location='cuda',weights_only=True)
 net=torch.nn.Sequential(torch.nn.Linear(169,512),torch.nn.ELU(),torch.nn.Linear(512,256),torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),torch.nn.Linear(128,12)).cuda().eval()
 net.load_state_dict({k.removeprefix('actor.'):v for k,v in ckpt['model_state_dict'].items() if k.startswith('actor.')})
 height=[];up=[];errors=[];trace=[]
 env.reset()
 with torch.inference_mode():
  for step in range(480):
   obs=env.get_observations()['policy'];assert obs.shape==(1,169)
   a=net(obs);assert torch.isfinite(a).all()
   env.step(a)
   pos=env.robot.get_pos()[0];quat=env.robot.get_quat()[0]
   q=env.robot.get_qpos()[0,7:];ref=env._current_ref_state()
   h=float(pos[2]);u=float(1-2*(quat[1]**2+quat[2]**2));err=float(torch.sqrt((q-ref['joint_rot'][0]).square().mean()))
   height.append(h);up.append(u);errors.append(err)
   trace.append({'teacher_observation_before_step':obs[0].cpu().tolist() if step<8 else None,'root_pos_after_step':pos.cpu().tolist(),'root_quat_after_step':quat.cpu().tolist(),'time':(step+1)/60,'height':h,'upright':u,'pose_rmse':err,'q':q.cpu().tolist(),'action':a[0].cpu().tolist()})
 row={'seed':int(os.environ['KINE_SCREEN_SEED']),'clip':clip,'source_checkpoint_sha256':hashlib.sha256((p/'logs/model.pt').read_bytes()).hexdigest(),'seconds':8,'control_hz':60,'physics_hz':240,'minimum_height':min(height),'minimum_upright':min(up),'mean_pose_rmse':float(np.mean(errors)),'upright_fraction':float(np.mean(np.array(up)>.5)),'functional_candidate':bool(np.mean(np.array(up)>.5)>.95 and np.mean(errors)<.6),'physical_qualified':False,'student_training_started':False}
 md=list(map(int,env.motor_dofs));assert sorted(md)==list(range(6,18));assert env.robot.get_qpos().shape[-1]==19;row['action_joint_names']=e['dof_names'];row['qpos_joint_names']=[name for _,name in sorted(zip(md,e['dof_names']))];row['motor_dof_indices']=md;row['teacher_input_layout']='31D time-major history x3 plus four future19D targets';row['configured_latency_seconds']=e['action_latency'];row['executed_latency_control_steps']=1 if e['action_latency']>0 else 0;rows.append(row);(OUTPUT/(clip+'_native_trace.json')).write_text(json.dumps(trace));(OUTPUT/('native_screen_'+clip+'.json')).write_text(json.dumps({'trials':[row],'capability_qualified':False},indent=2));print(row,flush=True)
 del env
(OUTPUT/'native_state.json').write_text(json.dumps({'phase':'screen_completed_pending_review','trials':len(rows),'student_training_started':False},indent=2))
