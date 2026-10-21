"""Training-engine diagnosis from a genuine four-foot reset; no rollout pose writes."""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--backflip-source',required=True)
p.add_argument('--unitree-source',required=True)
p.add_argument('--policy',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
p.add_argument('--seconds',default=8.,type=float)
p.add_argument('--seed',default=1000,type=int)
p.add_argument('--ground-contact-diagnostic',action='store_true')
AppLauncher.add_app_launcher_args(p)
a=p.parse_args();a.kit_args=(a.kit_args+' --enable isaacsim.asset.importer.urdf').strip()
app=AppLauncher(a).app
import json,torch
from isaaclab.envs import ManagerBasedRLEnv
from skills.sideflip.common import setup_sources,sha,write_json
from .task import make_cfg


def main():
 source=setup_sources(a.backflip_source,a.unitree_source)
 meta=json.loads(a.policy.with_suffix('.json').read_text())
 if sha(a.policy)!=meta['policy_sha256']:raise ValueError('Policy hash mismatch')
 cfg=make_cfg(source,1,a.seed,a.device or 'cuda:0')
 if meta.get('motor_model')=='dc_linear_v1':
     from skills.sideflip.common import enable_transfer_motors
     enable_transfer_motors(cfg,source)
 cfg.events.reset_base.params['upright_fraction']=0.
 cfg.observations.policy.enable_corruption=False
 cfg.commands.base_velocity.debug_vis=False
 cfg.commands.base_velocity.ranges.lin_vel_x=(.2,.2)
 cfg.commands.base_velocity.ranges.lin_vel_y=(0.,0.)
 cfg.commands.base_velocity.ranges.ang_vel_z=(0.,0.)
 cfg.commands.base_velocity.rel_standing_envs=0.
 for name in list(vars(cfg.terminations)):
  if not name.startswith('_'):setattr(cfg.terminations,name,None)
 ground_names=['base','Head_upper','Head_lower']+[leg+'_'+part for leg in ['FL','FR','RL','RR'] for part in ['hip','thigh','calf','foot']]
 if a.ground_contact_diagnostic:
  from isaaclab.sensors import ContactSensorCfg
  for name in ground_names:
   setattr(cfg.scene,'ground_'+name,ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/'+name,filter_prim_paths_expr=['/World/ground/terrain/GroundPlane/CollisionPlane'],history_length=1,debug_vis=False))
 cfg.episode_length_s=100.
 env=ManagerBasedRLEnv(cfg)
 try:
  obs,_=env.reset();robot=env.scene['robot'];sensor=env.scene['contact_forces']
  fronts=sensor.find_bodies('F.*_foot')[0];rears=sensor.find_bodies('R.*_foot')[0]
  foot_names=[leg+'_foot' for leg in ['FL','FR','RL','RR']]
  foot_ids=[robot.body_names.index(n) for n in foot_names]
  force_ids=[sensor.body_names.index(n) for n in foot_names]
  def forbidden(*args,**kw):raise RuntimeError('Unexpected root write during rollout')
  robot.write_root_state_to_sim=forbidden
  actor=torch.jit.load(str(a.policy),map_location=cfg.sim.device).eval()
  physics={}
  for name in ['get_masses','get_inertias','get_coms','get_dof_armatures','get_dof_friction_coefficients']:
   try:physics[name]=getattr(robot.root_physx_view,name)()[0].tolist()
   except Exception as exc:physics[name]={'unavailable':type(exc).__name__}
  physics['body_names']=robot.body_names
  if a.ground_contact_diagnostic:
   import omni.usd
   from pxr import UsdPhysics
   physics['floor_colliders']=[str(p.GetPath()) for p in omni.usd.get_context().get_stage().Traverse() if str(p.GetPath()).startswith('/World/ground') and p.HasAPI(UsdPhysics.CollisionAPI)]
   physics['ground_filter']='/World/ground/terrain/GroundPlane/CollisionPlane'
  trace=[]
  for i in range(round(a.seconds/env.step_dt)):
   input_obs=obs['policy'].clone()
   before_q=robot.data.joint_pos[0].tolist();before_dq=robot.data.joint_vel[0].tolist()
   with torch.inference_mode():act=actor(input_obs)
   obs,reward,terminated,truncated,info=env.step(act)
   if terminated.any() or truncated.any():raise RuntimeError('Unexpected episode reset')
   d=robot.data
   front=(torch.linalg.vector_norm(sensor.data.net_forces_w[0,fronts],dim=-1)>4).any()
   rear=(torch.linalg.vector_norm(sensor.data.net_forces_w[0,rears],dim=-1)>4).any()
   ground={}
   if a.ground_contact_diagnostic:
    for name in ground_names:
     matrix=env.scene['ground_'+name].data.force_matrix_w
     if matrix is None or matrix.numel()==0:raise RuntimeError('Missing filtered ground contact data')
     ground[name]=matrix[0].reshape(-1,3).sum(dim=0).tolist()
   trace.append(dict(joints=d.joint_pos[0].tolist(),feet_position=d.body_pos_w[0,foot_ids].tolist(),feet_velocity=d.body_lin_vel_w[0,foot_ids].tolist(),
    foot_forces=sensor.data.net_forces_w[0,force_ids].tolist(),ground_forces=ground,observation=input_obs[0].tolist(),action=act[0].tolist(),before_joints=before_q,before_joint_velocity=before_dq,time=(i+1)*env.step_dt,position=d.root_pos_w[0].tolist(),quaternion=d.root_quat_w[0].tolist(),
    projected_gravity=d.projected_gravity_b[0].tolist(),front_ground=bool(front),rear_ground=bool(rear),
    joint_velocity=d.joint_vel[0].tolist()))
  a.output.mkdir(parents=True,exist_ok=False)
  write_json(a.output/'trace.json',trace)
  write_json(a.output/'physics_contract.json',physics)
  tail=trace[-100:]
  error=lambda x:sum((u-v)**2 for u,v in zip(x['projected_gravity'],[-1,0,0]))**.5
  duration=sum(error(x)<.45 and x['position'][2]>.42 for x in trace)*env.step_dt
  travel=sum((tail[-1]['position'][i]-tail[0]['position'][i])**2 for i in [0,1])**.5
  support=sum(x['rear_ground'] and not x['front_ground'] for x in tail)/len(tail)
  steps={n:0 for n in ['RL_foot','RR_foot']};air={n:0. for n in steps};peak={n:0. for n in steps}
  stance_speeds=[];single_support=[]
  ground_z=float(env.scene.env_origins[0,2])
  for x in trace[-200:]:
   current=[]
   for name in steps:
    k=foot_names.index(name);f=x['ground_forces'].get(name,x['foot_forces'][k])
    contact=sum(v*v for v in f)**.5>4.;current.append(contact)
    height=x['feet_position'][k][2]-ground_z
    if contact:
     if height<=.035:
      if air[name]>=.04 and peak[name]>=.04 and error(x)<.45:steps[name]+=1
      air[name]=0.;peak[name]=0.
     if error(x)<.45:stance_speeds.append(sum(v*v for v in x['feet_velocity'][k][:2])**.5)
    else:
     air[name]+=env.step_dt;peak[name]=max(peak[name],height)
   if error(x)<.45:single_support.append(sum(current)==1)
  mean_stance=sum(stance_speeds)/max(1,len(stance_speeds))
  single_fraction=sum(single_support)/max(1,len(single_support))
  filtered_bad=None
  normal=None
  if a.ground_contact_diagnostic:
   normal=sum(abs(sum(f[2] for f in x['ground_forces'].values())) for x in tail)/len(tail)
   if normal<15.:raise RuntimeError('Ground filter failed support-force sanity check')
   filtered_bad=any(any(sum(v*v for v in f)**.5>1. for name,f in x['ground_forces'].items() if 'foot' not in name and 'calf' not in name) for x in trace)
  write_json(a.output/'summary.json',dict(backend='IsaacLab/PhysX',role='training-engine diagnosis only',
   policy_sha256=sha(a.policy),seed=a.seed,joint_names=robot.joint_names,foot_names=foot_names,started_on_four_feet=True,root_state_writes='initialization only; guarded during rollout',
   upright_seconds=duration,final_upright_travel_m=travel,rear_support_fraction=support,
   final_gravity=tail[-1]['projected_gravity'],max_height=max(x['position'][2] for x in trace),
   posture_and_travel_pass=duration>=2 and travel>.2 and support>=.8 and all(error(x)<.45 for x in tail),
   rear_recontact_counts_last4s=steps,mean_rear_stance_speed_m_s=mean_stance,single_rear_support_fraction_last4s=single_fraction,
   gait_validation_pass=None if filtered_bad is None else bool(duration>=2 and travel>.2 and support>=.8 and all(error(x)<.45 for x in tail) and all(n>=2 for n in steps.values()) and single_fraction>=.15 and mean_stance<=.15 and not filtered_bad),
   gait_criteria=dict(min_recontacts_per_rear_foot_last4s=2,min_air_s=.04,min_peak_link_height_m=.04,max_landing_link_height_m=.035,min_single_support_fraction=.15,max_mean_stance_link_speed_m_s=.15),
   filtered_nonfoot_ground_contact=filtered_bad,filtered_ground_normal_magnitude_N_last2s=normal,
   hardware_connected=False))
 finally:env.close()

try:main()
except Exception:
 import traceback,os,sys
 traceback.print_exc();sys.stderr.flush();os._exit(1)
finally:app.close(wait_for_replicator=False,skip_cleanup=True)
