"""Recovery function first; report physical constraints separately, never relabel teacher as student."""
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import rob2nav_recovery as task
from mjlab.envs import ManagerBasedRlEnv
from go2_supermimic.supermimic.train import Actor
from rob2nav_contact_audit import BodyContactAudit,forbidden_mask
ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--num-envs',type=int,default=128);ap.add_argument('--seconds',type=float,default=12);ap.add_argument('--output',required=True);args=ap.parse_args()
torch.manual_seed(args.seed);cfg=task.cfg(True);cfg.seed=args.seed;cfg.scene.num_envs=args.num_envs;cfg.episode_length_s=args.seconds+1
env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');obs,_=env.reset();c=task.st.robot_consts(env);robot=env.scene['robot'];limits=task.velocity_limits(env)
ck=torch.load(args.checkpoint,map_location=env.device,weights_only=False);assert ck['cfg']['task']==task.TASK
actor=Actor(obs['actor'].shape[-1],tuple(ck['cfg']['actor_hidden']),12).to(env.device);actor.load_state_dict(ck['actor']);actor.eval()
audit=BodyContactAudit(env);footids=[i for i,n in enumerate(audit.names) if n.endswith('_foot_collision')];assert len(footids)==4
lookup=torch.full((len(audit.names),),-1,device=env.device,dtype=torch.long);lookup[footids]=torch.arange(4,device=env.device)
N=env.num_envs;recovered=torch.zeros(N,dtype=torch.bool,device=env.device);stable=torch.zeros(N,device=env.device);first=torch.full((N,),-1.,device=env.device);peak_v=peak_q=peak_tau=peak_target=0.;samples=0
original=env.metrics_manager.compute_substep
def observer():
 global peak_v,peak_q,peak_tau,samples
 original();audit.update();q=robot.data.joint_pos;hard=robot.data.joint_pos_limits;peak_v=max(peak_v,float((robot.data.joint_vel.abs()/limits).max()));peak_q=max(peak_q,float(((hard[...,0]-q).clamp(min=0)+(q-hard[...,1]).clamp(min=0)).max()));tau=env.sim.data.qfrc_actuator[:,robot.indexing.joint_v_adr];peak_tau=max(peak_tau,float((tau.abs()/c.effort).max()));samples+=1
env.metrics_manager.compute_substep=observer;prev=robot.data.joint_pos.clone()
with torch.inference_mode():
 for i in range(round(args.seconds/env.step_dt)):
  action=actor(obs['actor']);target=c.default_q+action*c.action_scale;peak_target=max(peak_target,float(((target-prev).abs()/env.step_dt/limits).max()));prev=target.clone();obs,_,terminated,truncated,_=env.step(action);assert not bool((terminated|truncated).any())
  geom=audit.geom;world=audit.world;dist=audit.dist;valid=(torch.arange(len(geom),device=env.device)<audit.nacon[0])&(world>=0)&(world<N)&(geom[:,0]>=0)&(geom[:,1]>=0)&(dist<=0)
  a=geom[:,0].clamp(0,len(lookup)-1).long();b=geom[:,1].clamp(0,len(lookup)-1).long();foot=torch.where(audit.ground[a],lookup[b],torch.where(audit.ground[b],lookup[a],-1));m=valid&(foot>=0);contact=torch.zeros(N*4,device=env.device,dtype=torch.long);index=world[m].long()*4+foot[m];contact.scatter_add_(0,index,torch.ones_like(index));support=(contact.reshape(N,4)>0).sum(-1)
  forbidden=forbidden_mask(geom,world,dist,audit.nacon[0],audit.robot_nonfoot,audit.ground,N);bad=torch.zeros(N,device=env.device,dtype=torch.long);ids=world[forbidden].long();bad.scatter_add_(0,ids,torch.ones_like(ids))
  s=task.st.robot_state(env);good=(-s[:,task.st.GRAV][:,2]>.95)&(s[:,task.st.Z].squeeze(-1)>.24)&(s[:,task.st.V_YAW][:,:2].norm(dim=-1)<.2)&(s[:,task.st.W_B].norm(dim=-1)<.5)&(support>=3)&(bad==0)
  stable=torch.where(good,stable+env.step_dt,torch.zeros_like(stable));now=(stable>=2)&~recovered;first[now]=(i+1)*env.step_dt;recovered|=now
report={'seed':args.seed,'num_envs':N,'seconds':args.seconds,'checkpoint':args.checkpoint,'teacher_during_evaluation':False,'functional_successes':int(recovered.sum()),'functional_total':N,'functional_recovery_times_s':first.cpu().tolist(),'criterion':'upright>.95 height>.24 speed<.2 angvel<.5 at least3 actual foot collision contacts no nonfoot ground contact continuously2s; initial body contact allowed for recovery','physics_dt':env.physics_dt,'control_dt':env.step_dt,'physics_samples':samples,'max_joint_velocity_ratio':peak_v,'max_joint_limit_excess_rad':peak_q,'max_joint_torque_ratio':peak_tau,'max_target_rate_ratio':peak_target,'contact_audit':audit.summary(),'capability_qualified':False,'hardware_qualified':False}
p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(report,indent=2));print('RECOVERY_EVAL',json.dumps(report),flush=True);env.close()
