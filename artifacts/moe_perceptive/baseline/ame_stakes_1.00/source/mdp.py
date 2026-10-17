"""Explicit MoE observation/control order and terrain-aware training signals."""
import math
import torch
from .memory import ElevationMemory

JOINTS=[f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf')]
DEFAULT=[.1,.8,-1.5,-.1,.8,-1.5,.1,1.,-1.5,-.1,1.,-1.5]
FEET=[f'{leg}_foot' for leg in ('FL','FR','RL','RR')]


def reset_buffers(env,env_ids):
    if hasattr(env,'_moe_history'):
        env._moe_history[env_ids]=0;env._moe_history_step=-1
    if hasattr(env,'_moe_map'):env._moe_map.reset(env_ids)
    if hasattr(env,'_goal_hold'):env._goal_hold[env_ids]=0


def history(env):
    robot=env.scene['robot']
    if not hasattr(env,'_moe_history'):
        env._moe_history=torch.zeros(env.num_envs,5,45,device=env.device)
        env._moe_history_step=-1
        env._moe_joints=[robot.joint_names.index(n) for n in JOINTS]
        env._moe_default=torch.tensor(DEFAULT,device=env.device)
    if env._moe_history_step!=env.common_step_counter:
        command=env.command_manager.get_command('base_velocity')
        # A small simulation route supervisor asks the locomotion policy to stop
        # after reaching the target; it does not select footholds or edit actions.
        reached=(robot.data.root_pos_w-env.scene.env_origins)[:,0]>3.5
        command[reached]=0
        scales=torch.tensor([2.,2.,.25],device=env.device)
        obs=torch.cat((robot.data.root_ang_vel_b*.25,robot.data.projected_gravity_b,
            command*scales,robot.data.joint_pos[:,env._moe_joints]-env._moe_default,
            robot.data.joint_vel[:,env._moe_joints]*.05,env.action_manager.action),-1)
        if not torch.isfinite(obs).all():raise FloatingPointError('Nonfinite MoE proprioception')
        env._moe_history=torch.cat((env._moe_history[:,1:],obs[:,None]),1)
        env._moe_history_step=env.common_step_counter
    return env._moe_history.flatten(1)


def measured_map(env):
    r=env.scene['robot'];q=r.data.root_quat_w
    yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
    now=env.common_step_counter*env.step_dt
    if not hasattr(env,'_moe_map'):
        env._moe_map=ElevationMemory(env.num_envs,env.device);env._moe_map_frame=-1e6
    frame=now-env._moe_map_frame>=.099
    if frame:env._moe_map_frame=now
    fresh=(env.episode_length_buf==0)|frame
    hits=torch.cat([env.scene[n].data.ray_hits_w for n in ('depth_rays','lidar_rays')],1).clone()
    # No miss repair for outward-looking sensors: nonreturns are expected.
    if getattr(env.cfg,'moe_sensor_noise',False):
        hits[...,2]+=.005*torch.randn_like(hits[...,2])
        hits[torch.rand_like(hits[...,2])<.02]=torch.inf
    return env._moe_map.update(hits,r.data.root_pos_w,yaw,now,fresh)


def ideal_map(env):
    z=env.scene['height_scanner'].data.ray_hits_w[...,2]
    valid=torch.isfinite(z)
    height=(env.scene['robot'].data.root_pos_w[:,2,None]-z-.5).clamp(-1.5,2.)
    return torch.cat((torch.where(valid,height,0.),valid.float()),-1)


def forward_progress(env):return env.scene['robot'].data.root_lin_vel_w[:,0].clamp(-1,1.2)


def route_complete(env):
    p=env.scene['robot'].data.root_pos_w-env.scene.env_origins
    if not hasattr(env,'_goal_hold'):env._goal_hold=torch.zeros(env.num_envs,device=env.device)
    s=env.scene['contact_forces'];ids=[s.body_names.index(n) for n in FEET]
    stable=(s.data.net_forces_w[:,ids,2]>5.).sum(-1)>=2
    stable&=env.scene['robot'].data.projected_gravity_b[:,2]<-.9
    stable&=env.scene['robot'].data.root_lin_vel_b.norm(dim=-1)<.25
    target=(p[:,0]>3.5)&(p[:,1].abs()<.8)
    env._goal_hold=torch.where(target&stable,env._goal_hold+env.step_dt,0.)
    return env._goal_hold>=.3


def route_departure(env):
    p=env.scene['robot'].data.root_pos_w-env.scene.env_origins
    return (p[:,1].abs()>1.)|(p[:,0]<-.6)|(p[:,2]<-.35)


def lane_center(env):return (env.scene['robot'].data.root_pos_w[:,1]-env.scene.env_origins[:,1]).square()


def curriculum(env,env_ids):
    t=env.scene.terrain;p=env.scene['robot'].data.root_pos_w[env_ids]-env.scene.env_origins[env_ids]
    elapsed=env.episode_length_buf[env_ids]*env.step_dt
    up=(p[:,0]>3.4)&(p[:,1].abs()<.8)&~env.termination_manager.terminated[env_ids]&(elapsed>2.)
    down=env.termination_manager.terminated[env_ids]|(p[:,0]<1.5)
    # Keep a real capped hardest row instead of wrapping advanced cases to easy rows.
    t.terrain_levels[env_ids]=(t.terrain_levels[env_ids]+up.long()-(down&~up).long()).clamp(0,t.max_terrain_level-1)
    t.env_origins[env_ids]=t.terrain_origins[t.terrain_levels[env_ids],t.terrain_types[env_ids]]
    return t.terrain_levels.float().mean()


def landing_support(env):
    r=env.scene['robot'];sensor=env.scene['contact_forces']
    ids=[r.body_names.index(n) for n in FEET];contact_ids=[sensor.body_names.index(n) for n in FEET]
    foot_z=r.data.body_pos_w[:,ids,2]
    heights=torch.stack([env.scene['probe_'+n[:2]].data.ray_hits_w[...,2] for n in FEET],1)
    known=torch.isfinite(heights)
    supported=known&(heights>foot_z[:,:,None]-.08)&(heights<foot_z[:,:,None]+.04)
    margin=supported.float().mean(-1)
    first=sensor.compute_first_contact(env.step_dt)[:,contact_ids]
    moving=env.command_manager.get_command('base_velocity')[:,0]>.1
    return ((margin-.7)*first).sum(-1)*moving


def anchor_cost(env):
    if not hasattr(env,'_moe_anchor') or not hasattr(env,'_moe_history'):
        return torch.zeros(env.num_envs,device=env.device)
    with torch.no_grad():reference=env._moe_anchor(env._moe_history.flatten(1))
    error=((env.action_manager.action-reference)*.25).square().mean(-1)
    # Ground truth selects a training regularizer; it is never an actor input.
    ideal=env.scene['height_scanner'].data.ray_hits_w[...,2]
    valid=torch.isfinite(ideal)
    variation=torch.where(valid,ideal,-torch.inf).amax(-1)-torch.where(valid,ideal,torch.inf).amin(-1)
    return error*torch.where(valid.all(-1)&(variation<.05),4.,.2)


def match_joint_limits(robot):
    limits=robot.data.joint_pos_limits.clone()
    for leg in ('FL','FR','RL','RR'):
        pairs=[(-1.0472,1.0472),(-1.5708 if leg in ('FL','FR') else -.5236,1.5708),(-2.7227,-.83776)]
        for joint,pair in zip(('hip','thigh','calf'),pairs):
            limits[:,robot.joint_names.index(f'{leg}_{joint}_joint')]=torch.tensor(pair,device=robot.device)
    robot.write_joint_position_limit_to_sim(limits)
