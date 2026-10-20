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
 cfg=make_cfg(source,1,1000,a.device or 'cuda:0')
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
 cfg.episode_length_s=100.
 env=ManagerBasedRLEnv(cfg)
 try:
  obs,_=env.reset();robot=env.scene['robot'];sensor=env.scene['contact_forces']
  fronts=sensor.find_bodies('F.*_foot')[0];rears=sensor.find_bodies('R.*_foot')[0]
  def forbidden(*args,**kw):raise RuntimeError('Unexpected root write during rollout')
  robot.write_root_state_to_sim=forbidden
  actor=torch.jit.load(str(a.policy),map_location=cfg.sim.device).eval()
  trace=[]
  for i in range(round(a.seconds/env.step_dt)):
   with torch.inference_mode():act=actor(obs['policy'])
   obs,reward,terminated,truncated,info=env.step(act)
   if terminated.any() or truncated.any():raise RuntimeError('Unexpected episode reset')
   d=robot.data
   front=(torch.linalg.vector_norm(sensor.data.net_forces_w[0,fronts],dim=-1)>4).any()
   rear=(torch.linalg.vector_norm(sensor.data.net_forces_w[0,rears],dim=-1)>4).any()
   trace.append(dict(time=(i+1)*env.step_dt,position=d.root_pos_w[0].tolist(),quaternion=d.root_quat_w[0].tolist(),
    projected_gravity=d.projected_gravity_b[0].tolist(),front_ground=bool(front),rear_ground=bool(rear),
    joint_velocity=d.joint_vel[0].tolist()))
  tail=trace[-100:]
  error=lambda x:sum((u-v)**2 for u,v in zip(x['projected_gravity'],[-1,0,0]))**.5
  duration=sum(error(x)<.45 and x['position'][2]>.42 for x in trace)*env.step_dt
  travel=sum((tail[-1]['position'][i]-tail[0]['position'][i])**2 for i in [0,1])**.5
  support=sum(x['rear_ground'] and not x['front_ground'] for x in tail)/len(tail)
  a.output.mkdir(parents=True,exist_ok=False)
  write_json(a.output/'trace.json',trace)
  write_json(a.output/'summary.json',dict(backend='IsaacLab/PhysX',role='training-engine diagnosis only',
   policy_sha256=sha(a.policy),started_on_four_feet=True,root_state_writes='initialization only; guarded during rollout',
   upright_seconds=duration,final_upright_travel_m=travel,rear_support_fraction=support,
   final_gravity=tail[-1]['projected_gravity'],max_height=max(x['position'][2] for x in trace),
   posture_and_travel_pass=duration>=2 and travel>.2 and support>=.8 and all(error(x)<.45 for x in tail),
   hardware_connected=False))
 finally:env.close()

try:main()
except Exception:
 import traceback,os,sys
 traceback.print_exc();sys.stderr.flush();os._exit(1)
finally:app.close(wait_for_replicator=False,skip_cleanup=True)
