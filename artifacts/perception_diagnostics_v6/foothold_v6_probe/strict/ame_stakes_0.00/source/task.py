"""Mixed Go2 curriculum with visible-ray actor input and privileged critic."""
import copy,math
from isaaclab.utils import configclass
from isaaclab.managers import (ObservationGroupCfg,ObservationTermCfg,RewardTermCfg,
    TerminationTermCfg,CurriculumTermCfg,EventTermCfg,SceneEntityCfg)
from isaaclab.sensors import RayCasterCfg,patterns
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
from isaaclab.envs import mdp as lab_mdp
from perceptive_rl.terrain import LaneCfg
from perceptive_rl.sensors import add_sensors,CameraPattern
from . import mdp


def camera_pattern(cfg,device):
    import torch
    u=(torch.arange(64,device=device)+.5-32)/32*math.tan(math.radians(43.5))
    v=(torch.arange(32,device=device)+.5-16)/16*math.tan(math.radians(29.))
    vv,uu=torch.meshgrid(v,u,indexing='ij')
    d=torch.stack((torch.ones_like(uu),-uu,-vv),-1).reshape(-1,3)
    p=math.radians(getattr(cfg,'pitch_deg',20.));rotation=torch.tensor([[math.cos(p),0,math.sin(p)],[0,1,0],[-math.sin(p),0,math.cos(p)]],device=device)
    return torch.zeros_like(d),torch.nn.functional.normalize(d@rotation.T,dim=-1)


@configclass
class DenseCamera(CameraPattern):
    func=camera_pattern
    pitch_deg:float=20.


@configclass
class Policy(ObservationGroupCfg):
    history=ObservationTermCfg(func=mdp.history)
    visible_map=ObservationTermCfg(func=mdp.measured_map)
    def __post_init__(self):self.concatenate_terms=True;self.enable_corruption=False


@configclass
class Critic(Policy):
    ideal_map=ObservationTermCfg(func=mdp.ideal_map)
    velocity=ObservationTermCfg(func=lab_mdp.base_lin_vel)


@configclass
class Observations:
    policy:Policy=Policy()
    critic:Critic=Critic()


def make_cfg(count=512,seed=42,kind='mixed',evaluation=False,difficulty=None):
    cfg=UnitreeGo2RoughEnvCfg();cfg.seed=seed;cfg.scene.num_envs=count;cfg.scene.env_spacing=2.5
    cfg.episode_length_s=20.;cfg.moe_sensor_noise=not evaluation
    cfg.scene.robot.init_state.joint_pos=dict(zip(mdp.JOINTS,mdp.DEFAULT))
    cfg.scene.robot.actuators={
        'upper':IdealPDActuatorCfg(joint_names_expr=['.*_hip_joint','.*_thigh_joint'],stiffness=20.,damping=.5,
            effort_limit=23.7,effort_limit_sim=23.7,velocity_limit=30.,armature=.01,friction=.2),
        'calves':IdealPDActuatorCfg(joint_names_expr=['.*_calf_joint'],stiffness=20.,damping=.5,
            effort_limit=45.43,effort_limit_sim=45.43,velocity_limit=30.,armature=.01,friction=.2)}
    cfg.actions.joint_pos=JointPositionActionCfg(asset_name='robot',joint_names=mdp.JOINTS,
        preserve_order=True,scale=.25,use_default_offset=True)
    proportions={'flat':.125,'stairs':.25,'platform':.125,'ame_stakes':.25,'ame_alternate':.25}
    names=list(proportions) if kind=='mixed' else [kind]
    cfg.scene.terrain.terrain_generator=TerrainGeneratorCfg(seed=seed,size=(8.,8.),border_width=10.,
        num_rows=10,num_cols=8,horizontal_scale=.05,vertical_scale=.005,slope_threshold=.75,
        curriculum=True,use_cache=False,difficulty_range=(0.,1.) if difficulty is None else (difficulty,difficulty),
        sub_terrains={n:LaneCfg(kind=n,proportion=proportions.get(n,1.) if kind=='mixed' else 1.) for n in names})
    cfg.scene.terrain.max_init_terrain_level=0;cfg.scene.terrain.visual_material=None
    cfg.scene.sky_light.spawn.texture_file=None
    cfg.scene.height_scanner.offset.pos=(.25,0.,20.)
    cfg.scene.height_scanner.pattern_cfg=patterns.GridPatternCfg(resolution=.05,size=(1.6,1.),ordering='xy')
    cfg.scene.height_scanner.ray_alignment='yaw';cfg.scene.height_scanner.update_period=.02
    add_sensors(cfg);cfg.scene.depth_rays.pattern_cfg=DenseCamera()
    for name in mdp.FEET:
        setattr(cfg.scene,'probe_'+name[:2],RayCasterCfg(prim_path='{ENV_REGEX_NS}/Robot/'+name,
            update_period=.02,offset=RayCasterCfg.OffsetCfg(pos=(0.,0.,2.)),ray_alignment='yaw',
            pattern_cfg=patterns.GridPatternCfg(resolution=.03,size=(.06,.06)),mesh_prim_paths=['/World/ground']))
    cfg.observations=Observations()
    c=cfg.commands.base_velocity;c.debug_vis=False;c.heading_command=False;c.rel_heading_envs=0.
    c.rel_standing_envs=.05 if not evaluation else 0.;c.resampling_time_range=(20.,20.)
    c.ranges.lin_vel_x=(.3,.65) if not evaluation else (.5,.5)
    c.ranges.lin_vel_y=(0.,0.);c.ranges.ang_vel_z=(0.,0.)
    cfg.events.add_base_mass=None;cfg.events.base_com=None;cfg.events.push_robot=None
    cfg.events.base_external_force_torque=None
    cfg.events.reset_base.params={'pose_range':{'x':(-.03,.03),'y':(-.03,.03),'yaw':(-.04,.04)},'velocity_range':{}}
    cfg.events.reset_robot_joints.params={'position_range':(1.,1.),'velocity_range':(0.,0.)}
    cfg.events.physics_material.params.update(static_friction_range=(.8,.8),dynamic_friction_range=(.6,.6))
    cfg.events.moe_reset=EventTermCfg(func=mdp.reset_buffers,mode='reset')
    cfg.rewards.forward_progress=RewardTermCfg(func=mdp.forward_progress,weight=1.)
    cfg.rewards.lane_center=RewardTermCfg(func=mdp.lane_center,weight=-.5)
    cfg.rewards.landing_support=RewardTermCfg(func=mdp.landing_support,weight=1.)
    cfg.rewards.anchor_cost=RewardTermCfg(func=mdp.anchor_cost,weight=-2.)
    cfg.rewards.termination=RewardTermCfg(func=lab_mdp.is_terminated,weight=-10.)
    cfg.rewards.feet_air_time.weight=.2;cfg.rewards.feet_air_time.params['threshold']=.2
    cfg.rewards.dof_pos_limits.weight=-1.
    cfg.rewards.undesired_contacts=RewardTermCfg(func=lab_mdp.undesired_contacts,weight=-1.,
        params={'sensor_cfg':SceneEntityCfg('contact_forces',body_names=['.*_thigh','.*_calf']),'threshold':1.})
    cfg.terminations.route_complete=TerminationTermCfg(func=mdp.route_complete,time_out=True)
    cfg.terminations.route_departure=TerminationTermCfg(func=mdp.route_departure)
    cfg.terminations.bad_orientation=TerminationTermCfg(func=lab_mdp.bad_orientation,params={'limit_angle':1.})
    cfg.curriculum.terrain_levels=None if evaluation else CurriculumTermCfg(func=mdp.curriculum)
    return cfg
