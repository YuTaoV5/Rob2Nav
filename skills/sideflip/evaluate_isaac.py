"""Diagnostic playback in the training engine; not a MuJoCo acceptance substitute."""
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--backflip-source',required=True)
p.add_argument('--unitree-source',required=True)
p.add_argument('--motion',required=True,type=Path)
p.add_argument('--policy',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
p.add_argument('--seconds',default=6.,type=float)
p.add_argument('--seed',default=1000,type=int)
AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
args.kit_args=(args.kit_args+' --enable isaacsim.asset.importer.urdf').strip()
app=AppLauncher(args).app
import torch
import types
from isaaclab.envs import ManagerBasedRLEnv
from .common import setup_sources,sha,write_json
from .task import make_cfg


def main():
    source=setup_sources(args.backflip_source,args.unitree_source)
    contract=json.loads(args.motion.with_suffix('.json').read_text())
    meta=json.loads(args.policy.with_suffix('.json').read_text())
    if sha(args.policy)!=meta['policy_sha256'] or sha(args.motion)!=meta['reference_sha256']:
        raise ValueError('Policy/reference hash mismatch')
    cfg=make_cfg(source,args.motion,contract,1,args.seed,args.device or 'cuda:0')
    if meta.get('motor_model')=='dc_linear_v1':
        from skills.sideflip.common import enable_transfer_motors
        enable_transfer_motors(cfg,source)
    fixstand=meta.get('fixstand_roundtrip') or {}
    if fixstand:
        from skills.fixstand import configure_sideflip
        mode=fixstand['action_parameterization']
        configure_sideflip(cfg,contract,mode.startswith('reference_residual_'),
            fixstand['residual_limit_rad'],fixstand['bridge_residual_limit_rad'],
            fixstand['terminal_hip_taper_frames'],fixstand.get('boundary_blend_frames',0))
    cfg.observations.policy.enable_corruption=False
    cfg.events.base_com=None
    cfg.events.physics_material=None
    cfg.commands.motion.pose_range={}
    cfg.commands.motion.velocity_range={}
    cfg.commands.motion.joint_position_range=(-.002,.002)
    for name in list(vars(cfg.terminations)):
        if not name.startswith('_'):setattr(cfg.terminations,name,None)
    cfg.episode_length_s=100.
    from unitree_rl_lab.tasks.mimic.mdp.commands import MotionCommand
    original_sampling=MotionCommand._adaptive_sampling
    def start_at_zero(self,env_ids):self.time_steps[env_ids]=0
    MotionCommand._adaptive_sampling=start_at_zero
    env=ManagerBasedRLEnv(cfg)
    try:
        cmd=env.command_manager.get_term('motion')
        original_update=cmd._update_command
        def hold_endpoint(self):
            self.time_steps.clamp_(max=self.motion.time_step_total-2)
            original_update()
        cmd._update_command=types.MethodType(hold_endpoint,cmd)
        obs,_=env.reset()
        robot=env.scene['robot']
        # Any root write after initialization is an error, even if caused by the motion source.
        def forbidden(*a,**kw):raise RuntimeError('Unexpected root state write during diagnostic rollout')
        robot.write_root_state_to_sim=forbidden
        actor=torch.jit.load(str(args.policy),map_location=cfg.sim.device).eval()
        import numpy as np
        import xml.etree.ElementTree as ET
        from scipy.spatial.transform import Rotation
        limits={j.attrib['name']:j.find('limit').attrib for j in ET.parse(source/'resources/robots/go2/urdf/go2_description.urdf').getroot().findall('joint') if j.attrib['name'] in robot.joint_names}
        trace=[]
        for i in range(round(args.seconds/env.step_dt)):
            input_obs=obs['policy'].clone()
            with torch.inference_mode():act=actor(input_obs)
            obs,reward,terminated,truncated,info=env.step(act)
            if terminated.any() or truncated.any():raise RuntimeError('Unexpected episode reset')
            d=robot.data
            trace.append(dict(observation=input_obs[0].tolist(),action=act[0].tolist(),time=(i+1)*env.step_dt,position=d.root_pos_w[0].tolist(),
                quaternion=d.root_quat_w[0].tolist(),linear_velocity=d.root_lin_vel_w[0].tolist(),angular_velocity=d.root_ang_vel_b[0].tolist(),
                ground_bodies=[name for k,name in enumerate(env.scene['contact_forces'].body_names) if torch.linalg.vector_norm(env.scene['contact_forces'].data.net_forces_w[0,k]).item()>1.],
                joints=d.joint_pos[0].tolist(),joint_velocity=d.joint_vel[0].tolist(),
                projected_gravity=d.projected_gravity_b[0].tolist()))
        args.output.mkdir(parents=True,exist_ok=False)
        from isaaclab.utils.io import dump_yaml
        dump_yaml(str(args.output/'env.yaml'),cfg)
        write_json(args.output/'effective_fixstand.json',fixstand)
        write_json(args.output/'trace.json',trace)
        quat=np.array([x['quaternion'] for x in trace]);rot=Rotation.from_quat(quat[:,[1,2,3,0]])
        increments=(rot[:-1].inv()*rot[1:]).as_rotvec()
        turn=float(increments[:,0].sum());dominance=abs(turn)/max(1e-8,float(np.linalg.norm(increments,axis=1).sum()))
        tail=trace[-100:]
        airborne=0.;max_airborne=0.
        for x in trace:
            airborne=0. if x['ground_bodies'] else airborne+env.step_dt
            max_airborne=max(max_airborne,airborne)
        bad=any(any(not ('foot' in n or 'calf' in n) for n in x['ground_bodies']) for x in trace)
        speed_ratio=max(abs(v)/float(limits[n]['velocity']) for x in trace for n,v in zip(robot.joint_names,x['joint_velocity']))
        excess=max(0.,max(max(float(limits[n]['lower'])-v,v-float(limits[n]['upper'])) for x in trace for n,v in zip(robot.joint_names,x['joints'])))
        success=bool(5.5<abs(turn)<7.1 and dominance>.75 and max_airborne>=.08 and not bad and
            max(x['position'][2] for x in trace)>.45 and min(-x['projected_gravity'][2] for x in trace)<-.8 and
            all(-x['projected_gravity'][2]>.90 and x['position'][2]>.22 and np.linalg.norm(x['linear_velocity'][:2])<.25 and np.linalg.norm(x['angular_velocity'])<1. for x in tail))
        write_json(args.output/'summary.json',dict(args=dict(kind='sideflip',seed=args.seed),backend='IsaacLab/PhysX',
            role='native dynamics evaluation; not a MuJoCo result',successes=int(success),total=1,episodes=[dict(success=success)],
            policy_sha256=sha(args.policy),reference_sha256=sha(args.motion),frames=len(trace),
            initial_reference_frame=0,root_state_writes='episode initialization only; guarded during rollout',
            body_x_rotation_rad=turn,axis_dominance=dominance,airtime_s=max_airborne,unexpected_body_ground_contact=bad,
            max_height=max(x['position'][2] for x in trace),final_projected_gravity=trace[-1]['projected_gravity'],
            max_joint_speed_ratio=speed_ratio,max_joint_limit_excess_rad=excess,
            hardware_constraints_pass=speed_ratio<=1. and excess<=.001,hardware_connected=False,
            constraint_sampling_hz=1./env.step_dt,constraint_scope='control-frame diagnostic; not all physics substeps'))
    finally:
        env.close()
        MotionCommand._adaptive_sampling=original_sampling

try:main()
except Exception:
    import traceback,os,sys
    traceback.print_exc();sys.stderr.flush();os._exit(1)
finally:app.close(wait_for_replicator=False,skip_cleanup=True)
