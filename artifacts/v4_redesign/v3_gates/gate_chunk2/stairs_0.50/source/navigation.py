"""Goal-shaped sparse-terrain experiment, independently implemented from papers.

Privileged maps are an explicit teacher-only diagnostic/training input. Final
scores always use the original AME geometry, not the shallow-pit curriculum.
"""
import torch
from . import mdp


def goal(env):
    r=env.scene['robot'];q=r.data.root_quat_w
    yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
    rel=env.scene.env_origins[:,:2]+torch.tensor([3.6,0.],device=env.device)-r.data.root_pos_w[:,:2]
    x=rel[:,0]*yaw.cos()+rel[:,1]*yaw.sin();y=-rel[:,0]*yaw.sin()+rel[:,1]*yaw.cos()
    scale=2./rel.norm(dim=-1).clamp(min=2.)
    return torch.stack((x*scale,y*scale,-yaw.sin(),yaw.cos()),-1)


def teacher_map(env):
    ideal=mdp.ideal_map(env);h,valid=ideal[:,:693],ideal[:,693:]
    return torch.cat((h,valid,1-valid),-1)


def moving(env):
    r=env.scene['robot'];delta=env.scene.env_origins[:,:2]+torch.tensor([3.6,0.],device=env.device)-r.data.root_pos_w[:,:2]
    velocity=r.data.root_lin_vel_w[:,:2];speed=velocity.norm(dim=-1)
    aligned=(velocity*delta).sum(-1)>.5*speed*delta.norm(dim=-1)
    return ((delta.norm(dim=-1)<.2)|(aligned&(speed>.15)&(speed<1.5))).float()


def goal_proximity(env):
    delta=env.scene['robot'].data.root_pos_w[:,:2]-env.scene.env_origins[:,:2]-torch.tensor([3.6,0.],device=env.device)
    # End-of-episode shaping follows the navigation formulation in the papers.
    remaining=(env.max_episode_length-env.episode_length_buf)*env.step_dt
    return (1/(1+.25*delta.square().sum(-1)))*(remaining<4.)/4.


def stagnation(env):
    r=env.scene['robot'];now=env.common_step_counter
    if not hasattr(env,'_nav_progress'):
        env._nav_progress=r.data.root_pos_w[:,:2].clone();env._nav_progress_step=now
        env._nav_stagnated=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
    reset=env.episode_length_buf==0
    env._nav_progress[reset]=r.data.root_pos_w[reset,:2];env._nav_stagnated[reset]=False
    if now-env._nav_progress_step>=round(5./env.step_dt):
        distance=(r.data.root_pos_w[:,:2]-env._nav_progress).norm(dim=-1)
        to_goal=(r.data.root_pos_w[:,:2]-env.scene.env_origins[:,:2]-torch.tensor([3.6,0.],device=env.device)).norm(dim=-1)
        env._nav_stagnated=(distance<.25)&(to_goal>1.)&(env.episode_length_buf*env.step_dt>=5.)
        env._nav_progress=r.data.root_pos_w[:,:2].clone();env._nav_progress_step=now
    return env._nav_stagnated


def supported_contact(env):
    r=env.scene['robot'];s=env.scene['contact_forces'];ids=[s.body_names.index(n) for n in mdp.FEET]
    heights=torch.stack([env.scene['probe_'+n[:2]].data.ray_hits_w[...,2] for n in mdp.FEET],1)
    # Training-only scoring against original support elevation, not the raised
    # safety floor. This term does not select or correct a foot trajectory.
    surface=(heights-env.scene.env_origins[:,None,2,None]>-.055).float().mean(-1)
    first=s.compute_first_contact(env.step_dt)[:,ids]
    return ((surface-.7)*first).sum(-1)


def route_bonus(env):return env.termination_manager.get_term('route_complete').float()


def curriculum(env,env_ids):
    t=env.scene.terrain;p=env.scene['robot'].data.root_pos_w[env_ids]-env.scene.env_origins[env_ids]
    success=env.termination_manager.get_term('route_complete')[env_ids]
    if not hasattr(env,'_nav_success_ema'):env._nav_success_ema=torch.zeros(env.num_envs,device=env.device)
    env._nav_success_ema[env_ids]=.8*env._nav_success_ema[env_ids]+.2*success.float()
    up=success&(env._nav_success_ema[env_ids]>.5)
    down=p[:,0]<-.1  # Do not demote every failed first crossing back to level 0.
    t.terrain_levels[env_ids]=(t.terrain_levels[env_ids]+up.long()-down.long()).clamp(0,t.max_terrain_level-1)
    t.env_origins[env_ids]=t.terrain_origins[t.terrain_levels[env_ids],t.terrain_types[env_ids]]
    return t.terrain_levels.float().mean()


def apply(cfg,teacher=False,training=False):
    from isaaclab.managers import ObservationTermCfg,RewardTermCfg,TerminationTermCfg,CurriculumTermCfg
    cfg.moe_navigation=True;cfg.moe_teacher_map=teacher
    if teacher:cfg.observations.policy.visible_map=ObservationTermCfg(func=teacher_map)
    cfg.observations.policy.goal=ObservationTermCfg(func=goal)
    cfg.observations.critic.goal=ObservationTermCfg(func=goal)
    # Remove dense rewards for standing with zero yaw rate and fixed-speed
    # tracking. Let the policy choose its speed on sparse footholds.
    cfg.rewards.track_lin_vel_xy_exp=None;cfg.rewards.track_ang_vel_z_exp=None
    cfg.rewards.feet_air_time=None;cfg.rewards.landing_support=None
    cfg.rewards.forward_progress.weight=5.
    cfg.rewards.goal_motion=RewardTermCfg(func=moving,weight=5.)
    cfg.rewards.goal_position=RewardTermCfg(func=goal_proximity,weight=100.)
    cfg.rewards.route_bonus=RewardTermCfg(func=route_bonus,weight=500.)
    cfg.rewards.supported_contact=RewardTermCfg(func=supported_contact,weight=1.)
    cfg.rewards.termination.weight=-100.
    cfg.rewards.anchor_cost.weight=-.2
    cfg.terminations.stagnation=TerminationTermCfg(func=stagnation)
    if training:
        cfg.curriculum.terrain_levels=CurriculumTermCfg(func=curriculum)
        for terrain in cfg.scene.terrain.terrain_generator.sub_terrains.values():
            terrain.gentle_pit=True
    if teacher:
        # Teacher does not simulate an unavailable student sensor to save cost.
        cfg.scene.depth_rays=None;cfg.scene.lidar_rays=None
        cfg.observations.critic.visible_map=ObservationTermCfg(func=teacher_map)
