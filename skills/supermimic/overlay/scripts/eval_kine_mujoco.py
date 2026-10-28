"""Frozen Kine2Go teacher, real MuJoCo closed loop; no student or pose replay."""
import argparse,sys,json,time,re,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rob2nav_kine_teacher import diagnostic_cfg,KineMuJoCoTeacher
from go2_supermimic.supermimic import state as st
from mjlab.envs import ManagerBasedRlEnv
from rob2nav_contact_audit import BodyContactAudit
p=argparse.ArgumentParser();p.add_argument('--clip',default='solo8_crawl_slow');p.add_argument('--seed',type=int,default=36600);p.add_argument('--output-dir',required=True);a=p.parse_args()
out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
with (out/'started.json').open('x') as f:json.dump({'created_at':time.time(),'clip':a.clip,'seed':a.seed},f)
status={'phase':'initializing','student_training_started':False}
def save():
 t=out/'state.tmp';t.write_text(json.dumps(status,indent=2));t.replace(out/'state.json')
save()
try:
 torch.manual_seed(a.seed);cfg=diagnostic_cfg(a.clip);cfg.scene.num_envs=4;cfg.seed=a.seed
 env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');obs,_=env.reset();teacher=KineMuJoCoTeacher(env,a.clip);robot=env.scene['robot'];c=st.robot_consts(env)
 names=env.action_manager.get_term('joint_pos')._target_names
 limits=torch.tensor([next(v.velocity_limit for v in robot.cfg.articulation.actuators if any(re.fullmatch(x,n) for x in v.target_names_expr)) for n in names],device=env.device)
 body=BodyContactAudit(env);peaks={'velocity_ratio':0.,'joint_excess_rad':0.,'torque_ratio':0.,'target_rate_ratio':0.};samples=0
 original=env.metrics_manager.compute_substep
 def audit():
  global samples
  original();body.update();q=robot.data.joint_pos;lohi=robot.data.joint_pos_limits
  peaks['velocity_ratio']=max(peaks['velocity_ratio'],float((robot.data.joint_vel.abs()/limits).max()))
  peaks['joint_excess_rad']=max(peaks['joint_excess_rad'],float(((lohi[...,0]-q).clamp(min=0)+(q-lohi[...,1]).clamp(min=0)).max()))
  peaks['torque_ratio']=max(peaks['torque_ratio'],float((env.sim.data.qfrc_actuator[:,robot.indexing.joint_v_adr].abs()/c.effort).max()));samples+=1
 env.metrics_manager.compute_substep=audit
 initial={'root_position':robot.data.root_link_pos_w.cpu().tolist(),'root_quaternion':robot.data.root_link_quat_w.cpu().tolist(),'joint_position':robot.data.joint_pos.cpu().tolist(),'joint_velocity':robot.data.joint_vel.cpu().tolist()}
 upright=[];errors=[];traces=[];previous=None;status['phase']='running';status['teacher']=teacher.metadata;save()
 with torch.inference_mode():
  for i in range(480):
   action=teacher.act();target=c.default_q+action*c.action_scale
   if previous is not None:peaks['target_rate_ratio']=max(peaks['target_rate_ratio'],float(((target-previous).abs()/env.step_dt/limits).max()))
   previous=target.clone();_,_,done,truncated,_=env.step(action);assert not bool((done|truncated).any()),'Unexpected reset before eight seconds'
   teacher.after_step();ref=teacher.reference.at(teacher.clock,teacher.heading)
   up=-robot.data.projected_gravity_b[:,2];err=(robot.data.joint_pos[:,teacher.to_reference]-ref[:,7:]).square().mean(-1).sqrt()
   upright.append(up.cpu());errors.append(err.cpu())
   if i<8 or i%30==0:traces.append({'control_step':i+1,'upright':up.cpu().tolist(),'pose_rmse':err.cpu().tolist(),'current_native_action':teacher.current.cpu().tolist(),'executed_previous_action':action.cpu().tolist(),'root_position':(robot.data.root_link_pos_w-env.scene.env_origins).cpu().tolist()})
 u=torch.stack(upright);err=torch.stack(errors);rows=[{'env':j,'upright_fraction':float((u[:,j]>.5).float().mean()),'minimum_upright':float(u[:,j].min()),'mean_pose_rmse':float(err[:,j].mean()),'functional_candidate':bool((u[:,j]>.5).float().mean()>.95 and err[:,j].mean()<.6)} for j in range(4)]
 report={'teacher':teacher.metadata,'seed':a.seed,'seconds':8,'rows':rows,'functional_passed':sum(r['functional_candidate'] for r in rows),'functional_total':4,'physical':peaks,'body_contact':body.summary(),'physical_samples':samples,'expected_physical_samples':1920,'initial_state':initial,'trace':traces,'student_trained':False,'capability_qualified':False,'state_writes':'normal reset event only; never during rollout','source_sha256':{str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).resolve().parents[1]/'rob2nav_kine_teacher.py']}}
 assert samples==1920;(out/'report.json').write_text(json.dumps(report,indent=2));status.update(phase='completed_pending_review',functional_passed=report['functional_passed'],report=str(out/'report.json'));env.close()
except Exception as exc:status.update(phase='failed_requires_review',error=repr(exc))
status['updated_at']=time.time();save();print(json.dumps(status,indent=2),flush=True)
