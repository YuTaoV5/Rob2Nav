"""Teacher or student independent choreography evaluation; real physics only."""
import argparse,sys,json,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import rob2nav_expression as task
from rob2nav_moe_teacher import RoboGaugeTeacher
from mjlab.envs import ManagerBasedRlEnv
from go2_supermimic.supermimic.train import Actor
from go2_supermimic.supermimic import state as st
from rob2nav_contact_audit import BodyContactAudit
ap=argparse.ArgumentParser();ap.add_argument('--checkpoint');ap.add_argument('--seed',type=int,default=36110);ap.add_argument('--output',required=True);ap.add_argument('--num-envs',type=int,default=8);args=ap.parse_args()
torch.manual_seed(args.seed);cfg=task.cfg(True);cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed
# Each style is explicitly represented; after terminal reset retain style index.
env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');term=env.command_manager.get_term('target');resample=term._resample_command
def fixed(ids):
 resample(ids);term.gen.style[ids]=(ids%4).float();term.gen.speed_cap[ids]=4.
 term.gen.chase_vel[ids]=task.velocity_command(term.gen.wait[ids],term.gen.style[ids],term.gen.speed_cap[ids])
term._resample_command=fixed
obs,_=env.reset();robot=env.scene['robot'];c=st.robot_consts(env)
if args.checkpoint:
 ck=torch.load(args.checkpoint,map_location=env.device,weights_only=False);assert ck['cfg']['task']==task.TASK
 actor=Actor(350,tuple(ck['cfg']['actor_hidden']),12).to(env.device);actor.load_state_dict(ck['actor']);actor.eval();teacher=None
else:teacher=RoboGaugeTeacher(env,Path('/root/gpufree-data/Rob2Nav/deployment/skills/walk_moe.json'))
names=env.action_manager.get_term('joint_pos')._target_names
limits=torch.tensor([next(a.velocity_limit for a in robot.cfg.articulation.actuators if any(re.fullmatch(p,n) for p in a.target_names_expr)) for n in names],device=env.device)
body=BodyContactAudit(env);peaks={'velocity_ratio':0.,'joint_excess_rad':0.,'torque_ratio':0.,'target_rate_ratio':0.};samples=0
base=env.metrics_manager.compute_substep
def observe():
 global samples
 base();body.update();q=robot.data.joint_pos;hard=robot.data.joint_pos_limits
 peaks['velocity_ratio']=max(peaks['velocity_ratio'],float((robot.data.joint_vel.abs()/limits).max()))
 peaks['joint_excess_rad']=max(peaks['joint_excess_rad'],float(((hard[...,0]-q).clamp(min=0)+(q-hard[...,1]).clamp(min=0)).max()))
 peaks['torque_ratio']=max(peaks['torque_ratio'],float((env.sim.data.qfrc_actuator[:,robot.indexing.joint_v_adr].abs()/c.effort).max()));samples+=1
env.metrics_manager.compute_substep=observe
errors=[];actual=[];commands=[];falls=torch.zeros(args.num_envs,device=env.device);previous=None;clock_error=0.
with torch.inference_mode():
 for i in range(round(12/env.step_dt)):
  cmd=term.gen.chase_vel.clone();clock=term.gen.wait.clone()
  action=teacher.act(cmd) if teacher else actor(obs['actor']);target=c.default_q+action*c.action_scale
  if previous is not None:peaks['target_rate_ratio']=max(peaks['target_rate_ratio'],float(((target-previous).abs()/env.step_dt/limits).max()))
  previous=target.clone();obs,_,done,truncated,_=env.step(action)
  falls+=done.float()
  if teacher:teacher.reset(done|truncated)
  keep=~(done|truncated)
  if keep.any():clock_error=max(clock_error,float((term.gen.wait[keep]-clock[keep]-env.step_dt).abs().max()))
  s=st.robot_state(env);vel=torch.cat([s[:,st.V_YAW][:,:2],s[:,st.W_B][:,2:3]],-1)
  errors.append((vel-cmd).cpu());actual.append(vel.cpu());commands.append(cmd.cpu())
err=torch.stack(errors);vel=torch.stack(actual);cmd=torch.stack(commands);rows=[]
for style,label in enumerate(task.STYLES):
 mask=torch.arange(args.num_envs)%4==style
 rows.append({'style':label,'rmse_xy':float(err[:,mask,:2].square().sum(-1).mean().sqrt()),'rmse_yaw':float(err[:,mask,2].square().mean().sqrt()),'actual_velocity_spans':(vel[:,mask].amax((0,1))-vel[:,mask].amin((0,1))).tolist(),'falls':int(falls.cpu()[mask].sum())})
out={'role':'external_moe_plus_authored_rhythm_teacher' if teacher else 'own_supermimic_student','checkpoint':args.checkpoint,'seed':args.seed,'num_envs':args.num_envs,'seconds':12,'styles':rows,'physical':peaks,'body_contact':body.summary(),'physics_samples':samples,'expected_physics_samples':round(12/env.step_dt)*cfg.decimation,'clock_error_seconds':clock_error,'functional_candidate':bool(not falls.any() and all(r['rmse_xy']<.2 and r['rmse_yaw']<.25 for r in rows)),'emotion_perception_validated':False,'capability_qualified':False,'missing':['full uncut visual review','perceived expression review']}
assert samples==out['expected_physics_samples'];Path(args.output).write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True);env.close()
