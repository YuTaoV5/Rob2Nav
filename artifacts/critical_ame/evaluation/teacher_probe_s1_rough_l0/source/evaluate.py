"""Evaluate first episodes on the original upstream Go2 tasks, with video/export.

Launch: isaacsim/python.sh -m critical_ame.evaluate --upstream /path/to/upstream ...
The upstream network, rewards, terrain and joint contract are left unchanged.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument('--upstream',required=True)
p.add_argument('--checkpoint',required=True)
p.add_argument('--stage',type=int,choices=[1,2],required=True)
p.add_argument('--terrain',required=True)
p.add_argument('--level',type=int,default=0)
p.add_argument('--num-envs',type=int,default=32)
p.add_argument('--seed',type=int,default=801)
p.add_argument('--velocity',type=float,default=.5)
p.add_argument('--reset-jitter',type=float,default=0.,help='Additional validation XY reset range in metres; zero preserves upstream play reset')
p.add_argument('--yaw-jitter',type=float,default=0.,help='Additional validation reset yaw range in radians')
p.add_argument('--video',action='store_true')
p.add_argument('--repair-rays',action='store_true')
p.add_argument('--teacher-checkpoint',help='External diagnostic baseline only; never counted as an AME policy')
p.add_argument('--teacher-io')
p.add_argument('--output',required=True)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args()
if a.video:a.enable_cameras=True
upstream=Path(a.upstream).resolve()
for folder in ('rsl_rl','source/ame_locomotion','scripts/go2'):
    sys.path.insert(0,str(upstream/folder))
app=AppLauncher(a).app

import numpy as np
import torch
import gymnasium as gym
import ame_locomotion.tasks
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab.managers import TerminationTermCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from runner import Go2Runner
from export import ActorExport


def capture(e):
    robot=e.scene['robot'];sensor=e.scene['contact_forces']
    foot_ids=[i for i,n in enumerate(sensor.body_names) if n.endswith('_foot')]
    e._trial_position=(robot.data.root_pos_w-e.scene.env_origins).clone()
    e._trial_upright=-robot.data.projected_gravity_b[:,2].clone()
    e._trial_support=(sensor.data.net_forces_w[:,foot_ids,2]>5.).sum(-1)
    robot_foot_ids=[robot.body_names.index(sensor.body_names[i]) for i in foot_ids]
    e._trial_feet=(robot.data.body_pos_w[:,robot_foot_ids]-e.scene.env_origins[:,None,:]).clone()
    e._trial_foot_forces=sensor.data.net_forces_w[:,foot_ids].clone()
    return torch.zeros(e.num_envs,device=e.device,dtype=torch.bool)


torch.set_num_threads(4)
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
from .records import capture as capture_sources
capture_sources(out,a,upstream)
if a.repair_rays:
    from .repair import install
    install(out)
task=f'AME-Go2-Stage{a.stage}-Play-v0'
cfg=load_cfg_from_registry(task,'env_cfg_entry_point')
agent=load_cfg_from_registry(task,'rsl_rl_cfg_entry_point')
cfg.seed=a.seed;cfg.sim.device=a.device or 'cuda:0';cfg.scene.num_envs=a.num_envs
if a.reset_jitter or a.yaw_jitter:
    if min(a.reset_jitter,a.yaw_jitter)<0:raise ValueError('Jitter must be nonnegative')
    cfg.events.reset_base.params['pose_range']={'x':(-a.reset_jitter,a.reset_jitter),
        'y':(-a.reset_jitter,a.reset_jitter),'yaw':(-a.yaw_jitter,a.yaw_jitter)}
g=cfg.scene.terrain.terrain_generator
if a.terrain not in g.sub_terrains:raise ValueError(a.terrain)
g.sub_terrains={a.terrain:g.sub_terrains[a.terrain]};g.sub_terrains[a.terrain].proportion=1.
g.num_cols=1
if not 0<=a.level<g.num_rows:raise ValueError(a.level)
cfg.commands.base_velocity.ranges.lin_vel_x=(a.velocity,a.velocity)
cfg.terminations.capture_trial=TerminationTermCfg(func=capture)
if a.teacher_checkpoint:
    if not a.teacher_io:raise ValueError('External teacher needs --teacher-io')
    from .teacher import add_scanner
    add_scanner(cfg)
agent.device=cfg.sim.device;agent.seed=a.seed
env=gym.make(task,cfg=cfg,render_mode='rgb_array' if a.video else None)
e=env.unwrapped;t=e.scene.terrain
t.terrain_levels[:]=a.level;t.env_origins[:]=t.terrain_origins[t.terrain_levels,t.terrain_types]
e.scene.env_origins[:]=t.env_origins
wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
obs=wrapped.get_observations()
runner=Go2Runner(wrapped,agent.to_dict(),log_dir=None,device=cfg.sim.device)
runner.load(a.checkpoint,load_optimizer=False)
policy=runner.get_inference_policy()
if a.teacher_checkpoint:
    from .teacher import Teacher
    policy=Teacher(e,a.teacher_checkpoint,a.teacher_io)
robot=e.scene['robot'];action=e.action_manager.get_term('joint_pos')
initial_joint_std=float(robot.data.joint_pos.std(0,unbiased=False).mean())
actor=ActorExport(runner.alg.policy).cpu().eval()
if not a.teacher_checkpoint:
    with torch.inference_mode():
        example=obs['policy'].cpu()
        torch.testing.assert_close(actor(example),policy(obs).cpu(),atol=2e-5,rtol=2e-5)
        torch.jit.script(actor).save(str(out/'actor.pt'))
gains={name:{'kp':m.stiffness[0].tolist(),'kd':m.damping[0].tolist()} for name,m in robot.actuators.items()}
io={'observation_dim':2124,'joint_names':action._joint_names,'default_joint_pos':action._offset[0].tolist(),
    'action_scale':action._scale,'target_limits':action.limits[0].tolist(),'actuators':gains,
    'map':[21,33,3],'map_origin':'base yaw','proprio_scales':{'angular_velocity':.2,'joint_velocity':.05}}
(out/'io.json').write_text(json.dumps(io,indent=2))
done_first=torch.zeros(a.num_envs,device=e.device,dtype=torch.bool)
records=[];trace=[];frames=0;writer=None
if a.video:
    import imageio.v2 as imageio
    writer=imageio.get_writer(out/'episode_0.mp4',fps=25,macro_block_size=1)
try:
    with torch.inference_mode():
        for step in range(int(cfg.episode_length_s/e.step_dt)+5):
            actions=policy(obs)
            if not torch.isfinite(actions).all():raise FloatingPointError('nonfinite action')
            obs,reward,done,info=wrapped.step(actions)
            position=e._trial_position;support=e._trial_support;upright=e._trial_upright
            ended=done.bool() & ~done_first
            for i in ended.nonzero().flatten().tolist():
                complete=bool(e.termination_manager.get_term('tile_complete')[i])
                failed=bool(e.termination_manager.terminated[i])
                records.append({'env':i,'success':complete and not failed and int(support[i])>=2 and float(upright[i])>.7,
                    'route_complete':complete,'failure':failed,'support_feet':int(support[i]),
                    'upright':float(upright[i]),'position':position[i].cpu().tolist(),'seconds':(step+1)*e.step_dt,
                    'termination_terms':[name for name in e.termination_manager.active_terms
                        if bool(e.termination_manager.get_term(name)[i])]})
            if not done_first[0]:
                trace.append({'time':(step+1)*e.step_dt,'position':position[0].cpu().tolist(),
                              'support':int(support[0]),'upright':float(upright[0]),
                              'command':e.command_manager.get_command('base_velocity')[0].cpu().tolist(),
                              'feet':e._trial_feet[0].cpu().tolist(),
                              'foot_forces':e._trial_foot_forces[0].cpu().tolist(),
                              'action':actions[0].cpu().tolist()})
                if writer and step%2==0 and not ended[0]:
                    pos=robot.data.root_pos_w[0].cpu().numpy()
                    e.sim.set_camera_view(pos+np.array([2.2,2.5,1.7]),pos+np.array([.5,0.,0.]))
                    writer.append_data(e.render());frames+=1
            done_first|=ended
            if done_first.all():break
    result={'args':vars(a),'episodes':records,'successes':sum(r['success'] for r in records),'total':len(records),
        'policy_type':'external MLP diagnostic teacher' if a.teacher_checkpoint else 'upstream Go2 AME',
        'upstream_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip(),
        'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        'geometry_seed':g.seed,'video_frames':frames,'initial_joint_position_std':initial_joint_std,
        'note':'First episodes only; original play resets and fixed upstream terrain seed. Environments may share identical conditions.'}
    if a.teacher_checkpoint:
        result['teacher_sha256']=hashlib.sha256(Path(a.teacher_checkpoint).read_bytes()).hexdigest()
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'trace.json').write_text(json.dumps(trace))
    print('EVALUATION',result['successes'],'/',result['total'],flush=True)
finally:
    if writer:writer.close()
    env.close();app.close()
