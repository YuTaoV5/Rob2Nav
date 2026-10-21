"""Shared standard-stance training objective; never writes state outside episode resets."""
from pathlib import Path
import json

POSE = {leg+'_'+joint+'_joint':value for leg in ('FR','FL','RR','RL')
        for joint,value in [('hip',0.),('thigh',1. if leg.startswith('R') else .8),('calf',-1.5)]}


def target_pose(robot):
    import torch
    return torch.tensor([POSE[name] for name in robot.joint_names],device=robot.device)


def sideflip_boundary_limit(frame,total,hold_limit,bridge_limit,hold_frames=50,region_frames=90):
    """Blend stance compensation back to the original action near takeoff/landing."""
    import torch
    index=torch.as_tensor(frame)
    distance=torch.minimum(index,total-1-index)
    progress=torch.clamp((distance-hold_frames)/(region_frames-hold_frames),0.,1.)
    smooth=progress*progress*(3.-2.*progress)
    return hold_limit+(bridge_limit-hold_limit)*smooth


def terminal_hip_weight(frame,total,frames=50):
    """Keep calf support while phasing out only hip residuals near the final stand."""
    import torch
    if not 2<=frames<=50:raise ValueError('Hip taper must fit inside the final padded hold')
    x=torch.clamp((total-1-torch.as_tensor(frame))/(frames-1),0.,1.)
    return x*x*(3.-2.*x)


def stance_score(robot):
    import torch
    q=target_pose(robot)
    pose=torch.exp(-((robot.data.joint_pos-q)**2).mean(-1)/.01)
    velocity=torch.exp(-(robot.data.joint_vel**2).mean(-1)/1.)
    gravity=robot.data.projected_gravity_b-torch.tensor([0.,0.,-1.],device=robot.device)
    upright=torch.exp(-(gravity**2).sum(-1)/.02)
    still=torch.exp(-(robot.data.root_lin_vel_w**2).sum(-1)/.04-(robot.data.root_ang_vel_b**2).sum(-1)/.25)
    return pose*velocity*upright*still


def four_feet(env):
    import torch
    sensor=env.scene['contact_forces']
    ids=sensor.find_bodies('.*_foot')[0]
    if len(ids)!=4:raise RuntimeError('Expected exactly four Go2 feet')
    return (torch.linalg.vector_norm(sensor.data.net_forces_w[:,ids],dim=-1)>4.).all(-1).float()


def endpoint_reward(env, start=1., end=5.):
    t=env.episode_length_buf*env.step_dt
    mask=(t<start)|(t>=end)
    return mask*(stance_score(env.scene['robot'])+.25*four_feet(env))


def upright_phase(env):
    import torch
    t=env.episode_length_buf*env.step_dt
    # Hold / rise-and-upright / recover. Recovery command remains observable at t>=7.
    return torch.stack([(t<1.).float(),((t>=1.)&(t<5.)).float(),(t>=5.).float(),torch.clamp(t/8.,0.,1.)],dim=-1)


def upright_active(env):
    t=env.episode_length_buf*env.step_dt
    return ((t>=1.)&(t<5.)).float()


def upright_endpoint_reward(env):
    return endpoint_reward(env,start=1.,end=6.5)


def gated_upright_reward(env,original,parameters):
    return upright_active(env)*original(env,**parameters)


def upright_recovery_reward(env):
    import torch
    robot=env.scene['robot'];t=env.episode_length_buf*env.step_dt
    pose=torch.exp(-((robot.data.joint_pos-target_pose(robot))**2).mean(-1)/.15)
    error=robot.data.projected_gravity_b-torch.tensor([0.,0.,-1.],device=robot.device)
    gravity=torch.exp(-(error**2).sum(-1)/.3)
    return (t>=5.)*(pose+gravity+.25*four_feet(env))


def reset_standard(env,env_ids):
    import torch
    robot=env.scene['robot'];root=robot.data.default_root_state[env_ids].clone()
    root[:,:3]=env.scene.env_origins[env_ids];root[:,2]+=.32
    root[:,3:7]=torch.tensor([1.,0.,0.,0.],device=env.device);root[:,7:]=0.
    q=target_pose(robot).expand(len(env_ids),-1).clone()
    robot.write_root_state_to_sim(root,env_ids=env_ids)
    robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=env_ids)


def upright_endpoint_weight(time):
    import torch
    t=torch.as_tensor(time)
    release=torch.clamp((t-1.)/.5,0.,1.)
    recover=torch.clamp((t-5.)/1.5,0.,1.)
    smooth=lambda x:x*x*(3.-2.*x)
    return torch.maximum(1.-smooth(release),smooth(recover))


def configure_upright(cfg,endpoint_residual=False,endpoint_limit=.2):
    from isaaclab.managers import ObservationTermCfg,RewardTermCfg,EventTermCfg
    cfg.episode_length_s=8.
    cfg.events.reset_base=EventTermCfg(func=reset_standard,mode='reset')
    cfg.observations.policy.fixstand_phase=ObservationTermCfg(func=upright_phase)
    # Gate only task incentives; safety penalties remain active in all phases.
    for name in ('orientation','front_height','rear_support','front_contact','velocity','yaw','height',
                 'rear_slip','rear_airtime','single_support','rear_flight','rear_clearance'):
        term=getattr(cfg.rewards,name,None)
        if term is not None:
            term.params={'original':term.func,'parameters':dict(term.params)}
            term.func=gated_upright_reward
    cfg.rewards.fixstand_endpoints=RewardTermCfg(func=upright_endpoint_reward,weight=25.)
    cfg.rewards.fixstand_recovery=RewardTermCfg(func=upright_recovery_reward,weight=10.)
    cfg.commands.base_velocity.ranges.lin_vel_x=(0.,.15)
    cfg.commands.base_velocity.ranges.lin_vel_y=(0.,0.)
    cfg.commands.base_velocity.ranges.ang_vel_z=(0.,0.)
    if endpoint_residual:
        if not .01<=endpoint_limit<=.2:raise ValueError('Upright endpoint limit must be in [0.01, 0.2]')
        from .fixstand_actions import UprightEndpointAction
        cfg.actions.joint_pos.class_type=UprightEndpointAction
        cfg.actions.joint_pos.endpoint_residual_limit_rad=endpoint_limit
        cfg.commands.base_velocity.ranges.lin_vel_x=(.12,.20)
        cfg.commands.base_velocity.rel_standing_envs=0.


def configure_sideflip(cfg,contract,boundary_residual=False,residual_limit=.05,bridge_limit=None,terminal_hip_taper_frames=0):
    from isaaclab.managers import RewardTermCfg
    from skills.fixstand_motion import StartAtStand
    cfg.commands.motion.class_type=StartAtStand
    cfg.commands.motion.pose_range={}
    cfg.commands.motion.velocity_range={}
    cfg.commands.motion.joint_position_range=(0.,0.)
    duration=contract['frames']/contract['fps']
    cfg.episode_length_s=duration+.2
    cfg.rewards.fixstand_endpoints=RewardTermCfg(func=endpoint_reward,weight=25.,params={'start':1.,'end':duration-1.})
    if boundary_residual:
        if abs(contract['fps']-50.)>1e-6 or contract['frames']<=180:
            raise ValueError('Boundary residual contract requires 50Hz reference with 90-frame padded ends')
        from skills.fixstand_motion import FixstandReferenceAction
        cfg.actions.JointPositionAction.class_type=FixstandReferenceAction
        cfg.actions.JointPositionAction.residual_limit_rad=residual_limit
        cfg.actions.JointPositionAction.bridge_residual_limit_rad=residual_limit if bridge_limit is None else bridge_limit
        cfg.actions.JointPositionAction.terminal_hip_taper_frames=terminal_hip_taper_frames


def initialize_expanded_policy(runner,checkpoint):
    """Transfer matching layers; append zero columns for new phase observations, fresh optimizer."""
    import torch
    raw=torch.load(checkpoint,map_location=runner.device,weights_only=False)
    old=raw['model_state_dict'];model=runner.alg.policy;current=model.state_dict();expanded=[]
    for name,value in current.items():
        if name not in old:raise RuntimeError('Missing policy parameter '+name)
        source=old[name]
        if source.shape==value.shape:current[name]=source
        elif value.ndim==2 and source.ndim==2 and source.shape[0]==value.shape[0] and source.shape[1]+4==value.shape[1]:
            result=torch.zeros_like(value);result[:,:source.shape[1]]=source;current[name]=result;expanded.append(name)
        else:raise RuntimeError(f'Unsupported transfer shape {name}: {source.shape} -> {value.shape}')
    if not expanded:raise RuntimeError('Expected a phase-observation input expansion')
    model.load_state_dict(current)
    return {'initialization':'weight transfer; fresh optimizer and iteration counter','expanded_parameters':expanded}


def configure_backflip(cfg,velocity_limit_source=None,fast_safety_curriculum=False):
    import torch
    from isaaclab_backflip.tasks.go2_backflip.go2_backflip_env import Go2BackflipEnv
    cfg.episode_length_s=4.
    cfg.robot.init_state.joint_pos=dict(POSE)
    cfg.robot.init_state.pos=(0.,0.,.32)
    cfg.rewards.recovery_pose_error=.10
    cfg.rewards.success_hold_time_s=1.
    cfg.rewards.success_max_joint_speed=1.
    cfg.rewards.recovery_max_pitch_rate=.5
    cfg.rewards.scales['recovery_default_pose']=8.
    cfg.rewards.scales['recovery_still']=6.
    cfg.rewards.scales['recovery_dof_velocity']=-4.
    if velocity_limit_source is not None:
        from xml.etree import ElementTree as ET
        source=Path(velocity_limit_source)
        urdf=source/'resources/robots/go2/urdf/go2_description.urdf'
        limits={joint.attrib['name']:float(joint.find('limit').attrib['velocity'])
                for joint in ET.parse(urdf).getroot().findall('joint') if joint.find('limit') is not None}
        names=[f'{leg}_{joint}_joint' for leg in ('FL','FR','RL','RR') for joint in ('hip','thigh','calf')]
        cfg.control.joint_velocity_limits=tuple(limits[name] for name in names)
        if fast_safety_curriculum:
            cfg.control.velocity_termination_start_ratio=1.75
            cfg.rewards.safety_curriculum_start=.5
            cfg.rewards.safety_curriculum_warmup_steps=0
            cfg.rewards.safety_curriculum_ramp_steps=12000
    original=Go2BackflipEnv._get_rewards
    def rewards(self):
        base=original(self);t=self.episode_length_buf*self.step_dt
        delay=getattr(cfg,'entry_delay_seconds',0.)
        extra=((t<(.35+delay))|(t>=2.+delay))*stance_score(self._robot)*25.*self.step_dt
        self.extras.setdefault('log',{})['Fixstand/endpoint_score']=extra.mean()/self.step_dt
        return base+extra
    Go2BackflipEnv._get_rewards=rewards
    # The pinned upstream resets q=default, root orientation=identity and velocities=0.
    # Assert the default pose rather than hiding an incompatible asset reset.
    original_reset=Go2BackflipEnv._reset_idx
    def reset(self,env_ids):
        expected=target_pose(self._robot)
        if torch.max(torch.abs(self._robot.data.default_joint_pos[0]-expected))>1e-5:
            raise RuntimeError('Backflip default pose does not match the shared fixstand contract')
        return original_reset(self,env_ids)
    Go2BackflipEnv._reset_idx=reset


def backflip_entry_weight(time):
    import torch
    x=torch.clamp((torch.as_tensor(time)-1.)/.2,0.,1.)
    return 1.-x*x*(3.-2.*x)


def configure_backflip_delayed_entry(cfg,hold_scale=.4):
    """Train an observed one-second entry; raw actor -> bounded hold -> original control."""
    import torch
    if not 0.<=hold_scale<=.4:raise ValueError('Backflip hold scale must be in [0, 0.4]')
    from isaaclab_backflip.tasks.go2_backflip.go2_backflip_env import Go2BackflipEnv
    cfg.episode_length_s=5.
    cfg.entry_delay_seconds=1.
    original_pre=Go2BackflipEnv._pre_physics_step
    def pre(self,actions):
        raw_time=self.episode_length_buf.float()*self.step_dt
        w=backflip_entry_weight(raw_time).unsqueeze(-1)
        processed=torch.lerp(actions,hold_scale*torch.tanh(actions),w)
        return original_pre(self,processed)
    def phases(self):
        t=torch.clamp(self.episode_length_buf[:,None].float()*self.step_dt-1.,max=2.)
        phase=torch.pi*t/2.
        return tuple(f(phase/divisor) for divisor in (1.,2.,4.) for f in (torch.sin,torch.cos))
    Go2BackflipEnv._pre_physics_step=pre
    Go2BackflipEnv._phase_features=phases
    Go2BackflipEnv._time=lambda self:self.episode_length_buf.float()*self.step_dt-1.


def write_contract(path,skill,**extra):
    import hashlib
    Path(path).write_text(json.dumps(dict(skill=skill,objective='standard four-foot stance -> action -> standard four-foot stance',
        standard_pose=POSE,terminal_hold_s=1.,reset_only_state_writes=True,hardware_connected=False,
        shared_objective_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        independently_validated=False,**extra),ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def write_completion(folder,iterations):
    folder=Path(folder);checkpoints=list(folder.glob('model_*.pt'))
    if not checkpoints:raise RuntimeError('No checkpoint after training')
    last=max(checkpoints,key=lambda p:int(p.stem.split('_')[-1]))
    (folder/'completion.json').write_text(json.dumps(dict(execution_complete=True,requested_iterations=iterations,
        last_checkpoint=last.name,capability_verified=False,hardware_connected=False),indent=2)+'\n')
