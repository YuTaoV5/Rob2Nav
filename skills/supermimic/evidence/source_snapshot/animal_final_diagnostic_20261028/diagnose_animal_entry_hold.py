"""Teacher-OFF animal evaluation: reset perturbations and every 240Hz substep."""
import sys,argparse,json,hashlib,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import rob2nav_animal as task
from rob2nav_kine_reference import yaw_quaternion
from mjlab.envs import ManagerBasedRlEnv
from go2_supermimic.supermimic.train import Actor
from go2_supermimic.supermimic import state as st
from rob2nav_contact_audit import BodyContactAudit
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=36712);p.add_argument('--num-envs',type=int,default=128);p.add_argument('--joint-perturb-std',type=float,default=.025);a=p.parse_args()
output=Path(a.output);assert not output.exists(),'Refusing to overwrite an evaluation'
torch.manual_seed(a.seed);cfg=task.cfg(True);cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed
reset_delta=[]
def perturbed_reset(env,env_ids):
 task.reset_animal(env,env_ids)
 ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
 robot=env.scene['robot'];q=robot.data.joint_pos[ids].clone();qd=robot.data.joint_vel[ids].clone()
 delta=torch.randn_like(q)*a.joint_perturb_std
 robot.write_joint_state_to_sim(q+delta,qd,env_ids=ids);reset_delta.append(delta.cpu())
cfg.events['reset_reference'].func=perturbed_reset
env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');env._kine_intended_action=torch.zeros((a.num_envs,12),device=env.device);env._kine_previous_intended=torch.zeros_like(env._kine_intended_action);env._kine_reward_phase=torch.zeros(a.num_envs,device=env.device)
obs,_=env.reset();ck=torch.load(a.checkpoint,map_location=env.device,weights_only=False);assert ck['cfg']['task']==task.TASK
actor=Actor(350,tuple(ck['cfg']['actor_hidden']),12).to(env.device);actor.load_state_dict(ck['actor']);actor.eval();robot=env.scene['robot'];body=BodyContactAudit(env);c=st.robot_consts(env)
# Explicit alternate control/observation initialization, not a physics-state edit.
assert not cfg.observations['actor'].enable_corruption
initial_hold=(robot.data.joint_pos-c.default_q)/c.action_scale
env._kine_intended_action.copy_(initial_hold);env._kine_previous_intended.copy_(initial_hold)
env.observation_manager.reset(torch.arange(a.num_envs,device=env.device));obs=env.observation_manager.compute()
names=env.action_manager.get_term('joint_pos')._target_names
limits=torch.tensor([next(v.velocity_limit for v in robot.cfg.articulation.actuators if any(re.fullmatch(x,n) for x in v.target_names_expr)) for n in names],device=env.device)
def excess():
 q=robot.data.joint_pos;lohi=robot.data.joint_pos_limits
 return float(((lohi[...,0]-q).clamp(min=0)+(q-lohi[...,1]).clamp(min=0)).max())
initial_q=robot.data.joint_pos.cpu().clone();initial_qd=robot.data.joint_vel.cpu().clone()
diagnostic={'intended_peak_event':None,'maximum_intended_rate_after_first_second':0.,'contact_samples_first_quarter_second':0};actual_velocities=[];desired_velocities=[]
peaks={'velocity_ratio':float((robot.data.joint_vel.abs()/limits).max()),'joint_excess_rad':excess(),'torque_ratio':0.,'intended_target_rate_ratio':0.,'applied_target_rate_ratio':0.};samples=0
original=env.metrics_manager.compute_substep
def audit():
 global samples
 original();body.update();peaks['velocity_ratio']=max(peaks['velocity_ratio'],float((robot.data.joint_vel.abs()/limits).max()));peaks['joint_excess_rad']=max(peaks['joint_excess_rad'],excess())
 peaks['torque_ratio']=max(peaks['torque_ratio'],float((env.sim.data.qfrc_actuator[:,robot.indexing.joint_v_adr].abs()/c.effort).max()));samples+=1
 if samples==60:diagnostic['contact_samples_first_quarter_second']=body.summary()['nonfoot_ground_contact_samples']
env.metrics_manager.compute_substep=audit
ref,index,_,expected_names=task.parameters(env.device);assert list(robot.joint_names)==expected_names
heading=yaw_quaternion(torch.arange(a.num_envs,device=env.device)*torch.pi/2);start=(robot.data.root_link_pos_w-env.scene.env_origins).clone()
delay=initial_hold.clone();last_intent=delay.clone();last_applied=delay.clone();errors=[];uprights=[]
with torch.inference_mode():
 for i in range(480):
  action=actor(obs['actor']).clamp(-8,8)
  rates=((action-last_intent)*c.action_scale/env.step_dt/limits).abs();peak=float(rates.max())
  if diagnostic['intended_peak_event'] is None or peak>diagnostic['intended_peak_event']['ratio']:
   flat=int(rates.argmax());ei,ji=divmod(flat,12);diagnostic['intended_peak_event']={'control_step':i,'time_before_step':i*env.step_dt,'environment':ei,'joint':names[ji],'ratio':peak,'previous_target':float((c.default_q+last_intent*c.action_scale)[ei,ji]),'next_target':float((c.default_q+action*c.action_scale)[ei,ji])}
  if i>=60:diagnostic['maximum_intended_rate_after_first_second']=max(diagnostic['maximum_intended_rate_after_first_second'],peak)
  peaks['intended_target_rate_ratio']=max(peaks['intended_target_rate_ratio'],float(((action-last_intent)*c.action_scale/env.step_dt/limits).abs().max()))
  peaks['applied_target_rate_ratio']=max(peaks['applied_target_rate_ratio'],float(((delay-last_applied)*c.action_scale/env.step_dt/limits).abs().max()))
  last_intent=action.clone();last_applied=delay.clone()
  env._kine_previous_intended.copy_(env._kine_intended_action);env._kine_intended_action.copy_(action);env._kine_reward_phase=env.command_manager.get_term('target').gen.wait+1
  obs,_,done,truncated,_=env.step(delay);assert not bool((done|truncated).any()),'No reset allowed during eight-second evaluation';delay=action.clone()
  phase=env.command_manager.get_term('target').gen.wait
  ref0=ref.at(phase.long(),heading);ref1=ref.at(phase.long()+1,heading)
  desired_velocities.append(((ref1[:,:2]-ref0[:,:2])*60).cpu());actual_velocities.append(robot.data.root_link_lin_vel_w[:,:2].cpu())
  goal,_,_,_=task.desired(env.command_manager.get_term('target').gen.wait);errors.append((robot.data.joint_pos-goal).square().mean(-1).sqrt().cpu());uprights.append((-robot.data.projected_gravity_b[:,2]).cpu())
 end=robot.data.root_link_pos_w-env.scene.env_origins;zero=torch.zeros(a.num_envs,device=env.device,dtype=torch.long);last=torch.full_like(zero,480);desired_displacement=ref.at(last,heading)[:,:2]-ref.at(zero,heading)[:,:2]
 actual_displacement=end[:,:2]-start[:,:2];progress=(actual_displacement*desired_displacement).sum(-1)/desired_displacement.square().sum(-1).clamp(min=1e-6)
 u=torch.stack(uprights);e=torch.stack(errors);rows=[]
 for j in range(a.num_envs):
  posture=bool((u[:,j]>.5).float().mean()>.95 and e[:,j].mean()<.6)
  rows.append({'env':j,'upright_fraction':float((u[:,j]>.5).float().mean()),'mean_pose_rmse':float(e[:,j].mean()),'legacy_posture_gate':posture,'desired_planar_displacement':desired_displacement[j].tolist(),'actual_planar_displacement':actual_displacement[j].tolist(),'projected_progress_ratio':float(progress[j]),'locomotion_candidate':bool(posture and progress[j]>.5)})
assert samples==1920
actual_v=torch.stack(actual_velocities);desired_v=torch.stack(desired_velocities)
diagnostic['planar_velocity_rmse_all']=float((actual_v-desired_v).square().sum(-1).mean().sqrt())
diagnostic['planar_velocity_rmse_after_one_second']=float((actual_v[60:]-desired_v[60:]).square().sum(-1).mean().sqrt())
diagnostic['mean_actual_planar_speed']=float(actual_v.norm(dim=-1).mean());diagnostic['mean_desired_planar_speed']=float(desired_v.norm(dim=-1).mean())
diagnostic['contact_samples_after_quarter_second']=body.summary()['nonfoot_ground_contact_samples']-diagnostic['contact_samples_first_quarter_second']
# Counterfactual state audit, no simulator or actor state writes.
state=st.robot_state(env).detach();phase=env.command_manager.get_term('target').gen.wait
q0=ref.at(phase.long(),heading);q1=ref.at(phase.long()+1,heading);world_velocity=torch.cat(((q1[:,:2]-q0[:,:2])*60,torch.zeros((a.num_envs,1),device=env.device)),-1)
w,x,y,z=q0[:,3:7].unbind(-1);yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z));cv,sv=yaw.cos(),yaw.sin();velocity_yaw=torch.stack((cv*world_velocity[:,0]+sv*world_velocity[:,1],-sv*world_velocity[:,0]+cv*world_velocity[:,1]),-1)
stationary=state.clone();moving=state.clone();stationary[:,st.V_YAW][:,:2]=0;moving[:,st.V_YAW][:,:2]=velocity_yaw
old_stationary=task.reward(stationary,phase,env._kine_intended_action,env._kine_previous_intended,c.default_q,c.action_scale);old_moving=task.reward(moving,phase,env._kine_intended_action,env._kine_previous_intended,c.default_q,c.action_scale)
bonus_still=2*torch.exp(-velocity_yaw.square().sum(-1)/.25**2);bonus_moving=torch.full_like(bonus_still,2.)
diagnostic['old_reward_difference_same_posture_stationary_vs_reference_speed_max']=float((old_moving-old_stationary).abs().max())
diagnostic['proposed_velocity_bonus_reference_minus_stationary_mean']=float((bonus_moving-bonus_still).mean());diagnostic['proposed_bonus']={'weight':2.,'sigma_mps':.25,'trained_or_qualified':False}
diagnostic['interpretation']='Read-only response/entry diagnosis; velocity reward comparison does not establish the sole cause or prove improved control.' 
initial={'joint_position':initial_q,'joint_velocity':initial_qd,'reset_delta':reset_delta[-1],'seed':a.seed,'joint_perturb_std':a.joint_perturb_std};initial_path=output.with_suffix('.initial.pt');torch.save(initial,initial_path)
out={'initial_control_history_contract':'hold current q in delay buffer and intended-action observation; no qpos/qvel writes','diagnostic':diagnostic,'role':'own_student_teacher_OFF','checkpoint':a.checkpoint,'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),'seed':a.seed,'seconds':8,'num_envs':a.num_envs,'joint_perturb_std':a.joint_perturb_std,'perturbation_scope':'normal reset only, no rollout state writes','initial_joint_state_sha256':hashlib.sha256(initial_q.numpy().tobytes()+initial_qd.numpy().tobytes()).hexdigest(),'initial_tensor_sha256':hashlib.sha256(initial_path.read_bytes()).hexdigest(),'rows':rows,'locomotion_passed':sum(r['locomotion_candidate'] for r in rows),'locomotion_total':a.num_envs,'physical':peaks,'body_contact':body.summary(),'physical_samples':samples,'expected_physical_samples':1920,'locomotion_gate':'upright fraction >.95, mean pose RMSE <.6, projected desired displacement >50%','capability_qualified':False,'visual_review_complete':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
output.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='rows'},indent=2));env.close()
