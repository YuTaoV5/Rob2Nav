"""Velocity tracking evaluation; physics-substep joint audit, no training-state edits."""
import argparse,sys,json,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import rob2nav_velocity as v
from mjlab.envs import ManagerBasedRlEnv
from go2_supermimic.supermimic.train import Actor
from go2_supermimic.supermimic import state as st
ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",required=True);ap.add_argument("--seed",type=int,required=True)
ap.add_argument("--initialize-hold-targets",action="store_true");ap.add_argument("--audit-body-contact",action="store_true");ap.add_argument("--num-envs",type=int,default=128);ap.add_argument("--stage-seconds",type=float,default=10);ap.add_argument("--output",required=True)
args=ap.parse_args();torch.manual_seed(args.seed)
cfg=v.flat_cfg(True);cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed
cfg.commands["target"].resampling_time_range=(10000.,10000.)
if args.initialize_hold_targets:
 from rob2nav_reset_targets import install
 cfg=install(cfg)
env=ManagerBasedRlEnv(cfg=cfg,device="cuda:0");obs,_=env.reset()
robot=env.scene["robot"];term=env.command_manager.get_term("target");c=st.robot_consts(env)
ck=torch.load(args.checkpoint,map_location=env.device,weights_only=False)
if ck["cfg"]["task"]!="Go2-Velocity-Blind-Flat":raise ValueError("Wrong policy observation contract")
actor=Actor(obs["actor"].shape[-1],tuple(ck["cfg"]["actor_hidden"]),12).to(env.device)
actor.load_state_dict(ck["actor"]);actor.eval()
names=env.action_manager.get_term("joint_pos")._target_names
limits=[]
for name in names:
 matches=[a for a in robot.cfg.articulation.actuators if any(re.fullmatch(p,name) for p in a.target_names_expr)]
 assert len(matches)==1,name
 limits.append(matches[0].velocity_limit)
velocity_limits=torch.tensor(limits,device=env.device)
N=args.num_envs;dev=env.device
peak_v=torch.zeros(N,device=dev);peak_q=torch.zeros(N,device=dev);peak_tau=torch.zeros(N,device=dev)
from rob2nav_contact_audit import BodyContactAudit
body_audit=BodyContactAudit(env) if args.audit_body_contact else None
physics_samples=0
peak_event={}
original=env.metrics_manager.compute_substep
def observer():
 global physics_samples,peak_event
 original()
 if body_audit is not None:body_audit.update()
 # These fields directly index post-integration qpos/qvel; no observation refresh/reset.
 q=robot.data.joint_pos;qd=robot.data.joint_vel;tau=env.sim.data.qfrc_actuator[:,robot.indexing.joint_v_adr]
 hard=robot.data.joint_pos_limits
 ratios=qd.abs()/velocity_limits
 if float(ratios.max())>float(peak_v.max()):
  flat=int(ratios.argmax());ei=flat//12;ji=flat%12
  peak_event={"stage":label,"sample":physics_samples,"time":physics_samples*env.physics_dt,"environment":ei,"joint":names[ji],"velocity":float(qd[ei,ji]),"velocity_limit":float(velocity_limits[ji]),"command":list(command)}
 peak_v.copy_(torch.maximum(peak_v,ratios.max(-1).values))
 excess=((hard[...,0]-q).clamp(min=0)+(q-hard[...,1]).clamp(min=0)).max(-1).values
 peak_q.copy_(torch.maximum(peak_q,excess))
 peak_tau.copy_(torch.maximum(peak_tau,(tau.abs()/c.effort).max(-1).values))
 physics_samples+=1
env.metrics_manager.compute_substep=observer
stages=[("stand",(0.,0.,0.)),("forward",(.6,0.,0.)),("backward",(-.3,0.,0.)),
 ("left",(0.,.25,0.)),("right",(0.,-.25,0.)),("turn_left",(0.,0.,.6)),
 ("turn_right",(0.,0.,-.6)),("stop",(0.,0.,0.))]
falls=torch.zeros(N,device=dev);peak_target_rate=torch.zeros(N,device=dev)
previous_target=None;rows=[]
target_rate_event={}
fixed_command=torch.zeros((N,3),device=dev)
original_resample=term._resample_command
def fixed_resample(ids):
 original_resample(ids)
 term.gen.chase_vel[ids]=fixed_command[ids]
term._resample_command=fixed_resample
with torch.inference_mode():
 for label,command in stages:
  cmd=torch.tensor(command,device=dev).repeat(N,1)
  fixed_command.copy_(cmd)
  # Seed command observation before applying this stage, then collect normal history.
  term.gen.chase_vel.copy_(cmd);term._update_command(None)
  obs=env.observation_manager.compute(update_history=True)
  error_xy=torch.zeros(N,device=dev);error_yaw=torch.zeros(N,device=dev);count=0
  for i in range(round(args.stage_seconds/env.step_dt)):
   term.gen.chase_vel.copy_(cmd)
   a=actor(obs["actor"]);target=c.default_q+a*c.action_scale
   if previous_target is not None:
    rate=(target-previous_target).abs()/env.step_dt/velocity_limits
    if float(rate.max())>float(peak_target_rate.max()):
     flat=int(rate.argmax());ei=flat//12;ji=flat%12
     target_rate_event={"stage":label,"control_index":i,"time_in_stage":i*env.step_dt,
       "environment":ei,"joint":names[ji],"ratio":float(rate[ei,ji]),"command":list(command),
       "previous_target":float(previous_target[ei,ji]),"target":float(target[ei,ji])}
    peak_target_rate.copy_(torch.maximum(peak_target_rate,rate.max(-1).values))
   previous_target=target.clone()
   obs,_,terminated,_,_=env.step(a);falls+=terminated.float()
   if i*env.step_dt>=min(1.,args.stage_seconds/4):
    s=st.robot_state(env);error_xy+=(s[:,st.V_YAW][:,:2]-cmd[:,:2]).square().sum(-1)
    error_yaw+=(s[:,st.W_B][:,2]-cmd[:,2]).square();count+=1
  rows.append({"stage":label,"command":command,"rmse_xy":float((error_xy/count).mean().sqrt()),
               "rmse_yaw":float((error_yaw/count).mean().sqrt())})
raw={"initialize_hold_targets":args.initialize_hold_targets,"seed":args.seed,"checkpoint":args.checkpoint,"num_envs":N,"stages":rows,
 "physics_dt":env.physics_dt,"control_dt":env.step_dt,"decimation":cfg.decimation,
 "physics_sample_count":physics_samples,"expected_physics_samples":round(args.stage_seconds/env.step_dt)*len(stages)*cfg.decimation,
 "joint_velocity_peak_event":peak_event,"falls_total":int(falls.sum()),"falls_per_env_minute":float(falls.mean())/(len(stages)*args.stage_seconds/60),
 "max_joint_velocity_ratio":float(peak_v.max()),"max_joint_limit_excess_rad":float(peak_q.max()),
 "max_joint_torque_ratio":float(peak_tau.max()),"max_target_rate_ratio":float(peak_target_rate.max()),"target_rate_peak_event":target_rate_event,
 "actor_height_scan":False,"actor_lidar":False,"strict_validation_complete":False,
 "body_contact_audit":body_audit.summary() if body_audit else None,
 "missing_checks":([] if body_audit else ["non-foot body contact"])+["uncut original-speed video review"],
 "capability_qualified":False}
assert physics_samples==raw["expected_physics_samples"]
Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(raw,indent=2))
print("VELOCITY_EVALUATION",json.dumps(raw),flush=True);env.close()

