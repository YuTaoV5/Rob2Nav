"""Teacher-OFF animal student, full eight-second MuJoCo rollout without resets."""
import sys,argparse,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import rob2nav_animal as task
from rob2nav_kine_reference import yaw_quaternion
from mjlab.envs import ManagerBasedRlEnv
from go2_supermimic.supermimic.train import Actor
from rob2nav_contact_audit import BodyContactAudit
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=36710);a=p.parse_args()
torch.manual_seed(a.seed);cfg=task.cfg(True);cfg.scene.num_envs=4;cfg.seed=a.seed
env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');env._kine_intended_action=torch.zeros((4,12),device=env.device);env._kine_previous_intended=torch.zeros_like(env._kine_intended_action);env._kine_reward_phase=torch.zeros(4,device=env.device)
obs,_=env.reset();ck=torch.load(a.checkpoint,map_location=env.device,weights_only=False);assert ck['cfg']['task']==task.TASK
actor=Actor(350,tuple(ck['cfg']['actor_hidden']),12).to(env.device);actor.load_state_dict(ck['actor']);actor.eval();robot=env.scene['robot'];body=BodyContactAudit(env)
original=env.metrics_manager.compute_substep
def audit():original();body.update()
env.metrics_manager.compute_substep=audit
ref,index,_,names=task.parameters(env.device);assert list(robot.joint_names)==names
heading=yaw_quaternion(torch.arange(4,device=env.device)*torch.pi/2);start=robot.data.root_link_pos_w-env.scene.env_origins;start=start.clone()
delay=torch.zeros((4,12),device=env.device);errors=[];uprights=[]
with torch.inference_mode():
 for i in range(480):
  action=actor(obs['actor']).clamp(-8,8);env._kine_previous_intended.copy_(env._kine_intended_action);env._kine_intended_action.copy_(action);env._kine_reward_phase=env.command_manager.get_term('target').gen.wait+1
  obs,_,done,truncated,_=env.step(delay);assert not bool((done|truncated).any());delay=action.clone()
  goal,_,_,_=task.desired(env.command_manager.get_term('target').gen.wait);errors.append((robot.data.joint_pos-goal).square().mean(-1).sqrt().cpu());uprights.append((-robot.data.projected_gravity_b[:,2]).cpu())
 end=robot.data.root_link_pos_w-env.scene.env_origins;zero=torch.zeros(4,device=env.device,dtype=torch.long);last=torch.full_like(zero,480);desired_displacement=ref.at(last,heading)[:,:2]-ref.at(zero,heading)[:,:2]
 actual_displacement=end[:,:2]-start[:,:2];progress=(actual_displacement*desired_displacement).sum(-1)/desired_displacement.square().sum(-1).clamp(min=1e-6)
 u=torch.stack(uprights);e=torch.stack(errors);rows=[]
 for j in range(4):
  posture=bool((u[:,j]>.5).float().mean()>.95 and e[:,j].mean()<.6)
  rows.append({'env':j,'upright_fraction':float((u[:,j]>.5).float().mean()),'mean_pose_rmse':float(e[:,j].mean()),'legacy_posture_gate':posture,'desired_planar_displacement':desired_displacement[j].tolist(),'actual_planar_displacement':actual_displacement[j].tolist(),'projected_progress_ratio':float(progress[j]),'locomotion_candidate':bool(posture and progress[j]>.5)})
out={'role':'own_student_teacher_OFF','checkpoint':a.checkpoint,'seed':a.seed,'seconds':8,'num_envs':4,'rows':rows,'body_contact':body.summary(),'locomotion_gate':'legacy upright/pose gate plus >50% projected desired planar displacement; guards stationary pose-only false positives','capability_qualified':False,'purpose':'independent smoke-model diagnosis, not formal four-seed qualification'}
Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));env.close()
