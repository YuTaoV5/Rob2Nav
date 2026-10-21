"""Go2 reward/reset contract; ground-plane commands stay horizontal when the body stands up."""
import torch
from isaaclab.managers import RewardTermCfg as RewTerm, EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply
from isaaclab_tasks.manager_based.locomotion.velocity import mdp


def orientation_error(env):
    target = torch.tensor([-1.,0.,0.], device=env.device)
    return torch.sum((env.scene['robot'].data.projected_gravity_b-target)**2, dim=1)


def upright(env):
    return torch.exp(-orientation_error(env)/.35)


def front_height(env):
    robot = env.scene['robot']
    ids = robot.find_bodies('F.*_foot')[0]
    return torch.exp(-torch.mean((robot.data.body_pos_w[:,ids,2]-.65)**2, dim=1)/.04)


def rear_support(env):
    sensor = env.scene['contact_forces']
    ids = sensor.find_bodies('R.*_foot')[0]
    force = torch.linalg.vector_norm(sensor.data.net_forces_w[:,ids], dim=-1)
    return torch.mean((force > 4.).float(), dim=1)


def front_contact(env):
    sensor = env.scene['contact_forces']
    ids = sensor.find_bodies('F.*_foot')[0]
    return torch.mean((torch.linalg.vector_norm(sensor.data.net_forces_w[:,ids],dim=-1)>4.).float(),dim=1)


def horizontal_velocity(env):
    robot = env.scene['robot']
    # At pitch=-pi/2, body X points upward. Forward walking is along projected -body Z.
    minus_z = torch.zeros(env.num_envs,3,device=env.device)
    minus_z[:,2] = -1.
    forward = quat_apply(robot.data.root_quat_w, minus_z)[:,:2]
    norm = torch.linalg.vector_norm(forward, dim=1, keepdim=True)
    forward = forward/torch.clamp(norm,min=.05)
    side = torch.stack([-forward[:,1],forward[:,0]],dim=1)
    vel = robot.data.root_lin_vel_w[:,:2]
    return torch.stack([torch.sum(vel*forward,dim=1),torch.sum(vel*side,dim=1)],dim=1)


def track_velocity(env, std2=.20):
    target = env.command_manager.get_command('base_velocity')[:,:2]
    return upright(env)*torch.exp(-torch.sum((horizontal_velocity(env)-target)**2,dim=1)/std2)


def track_yaw(env):
    target = env.command_manager.get_command('base_velocity')[:,2]
    return upright(env)*torch.exp(-(env.scene['robot'].data.root_ang_vel_w[:,2]-target)**2/.25)


def height(env, target=.56, std2=.025):
    return upright(env)*torch.exp(-(env.scene['robot'].data.root_pos_w[:,2]-target)**2/std2)


def reset_mixed(env, env_ids, upright_fraction=.5):
    """Only episode reset writes state; actual rise/walk evaluation starts on four feet."""
    robot = env.scene['robot']
    count = len(env_ids)
    up = torch.rand(count,device=env.device) < upright_fraction
    root = robot.data.default_root_state[env_ids].clone()
    root[:,:3] = env.scene.env_origins[env_ids]
    root[:,:2] += torch.empty(count,2,device=env.device).uniform_(-.1,.1)
    root[:,2] += torch.where(up,.56,.32)
    yaw = torch.empty(count,device=env.device).uniform_(-.15,.15)
    pitch = torch.where(up,torch.full_like(yaw,-torch.pi/2),torch.zeros_like(yaw))
    root[:,3:7] = quat_from_euler_xyz(torch.zeros_like(yaw),pitch,yaw)
    root[:,7:] = 0.
    q = robot.data.default_joint_pos[env_ids].clone()
    for leg in ('RL','RR'):
        thigh = robot.find_joints(leg+'_thigh_joint')[0][0]
        calf = robot.find_joints(leg+'_calf_joint')[0][0]
        q[:,thigh] = torch.where(up,2.,q[:,thigh])
        q[:,calf] = torch.where(up,-1.,q[:,calf])
    for leg in ('FL','FR'):
        thigh = robot.find_joints(leg+'_thigh_joint')[0][0]
        q[:,thigh] = torch.where(up,.6,q[:,thigh])
    robot.write_root_state_to_sim(root,env_ids=env_ids)
    robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=env_ids)


@configclass
class UprightRewards:
    orientation = RewTerm(func=upright,weight=8.)
    front_height = RewTerm(func=front_height,weight=8.)
    rear_support = RewTerm(func=rear_support,weight=2.)
    front_contact = RewTerm(func=front_contact,weight=-2.)
    velocity = RewTerm(func=track_velocity,weight=3.)
    yaw = RewTerm(func=track_yaw,weight=1.)
    height = RewTerm(func=height,weight=2.)
    torque = RewTerm(func=mdp.joint_torques_l2,weight=-2e-5)
    joint_acc = RewTerm(func=mdp.joint_acc_l2,weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2,weight=-.01)
    joint_limit = RewTerm(func=mdp.joint_pos_limits,weight=-10.)


def make_cfg(backflip_source, num_envs, seed, device):
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
    from skills.sideflip.common import robot_config
    cfg = UnitreeGo2FlatEnvCfg()
    cfg.seed = seed
    cfg.sim.device = device
    cfg.scene.num_envs = num_envs
    cfg.scene.robot = robot_config(backflip_source)
    cfg.scene.terrain.visual_material = None
    cfg.scene.contact_forces.track_air_time = True
    cfg.sim.dt = .005
    cfg.decimation = 4
    cfg.sim.render_interval = 4
    cfg.actions.joint_pos.scale = .5
    cfg.rewards = UprightRewards()
    cfg.episode_length_s = 10.
    cfg.events.reset_base = EventTerm(func=reset_mixed,mode='reset',params={'upright_fraction':.5})
    cfg.events.reset_robot_joints = None
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.events.add_base_mass = None
    cfg.terminations.base_contact.params['sensor_cfg'] = SceneEntityCfg('contact_forces',body_names=['base','Head_.*'])
    cfg.commands.base_velocity.ranges.lin_vel_x = (-.3,.5)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-.15,.15)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-.5,.5)
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_standing_envs = .2
    cfg.commands.base_velocity.resampling_time_range = (3.,6.)
    return cfg


def rear_gait_state(env):
    robot=env.scene['robot'];sensor=env.scene['contact_forces']
    if not hasattr(env,'_upright_rear_ids'):
        env._upright_rear_ids=robot.find_bodies('R.*_foot')[0]
        env._upright_sensor_ids=sensor.find_bodies('R.*_foot')[0]
    contacts=torch.linalg.vector_norm(sensor.data.net_forces_w[:,env._upright_sensor_ids],dim=-1)>4.
    moving=torch.linalg.vector_norm(env.command_manager.get_command('base_velocity')[:,:2],dim=-1)>.08
    return robot,sensor,contacts,moving


def rear_stance_slip(env):
    robot,sensor,contacts,moving=rear_gait_state(env)
    velocity=robot.data.body_lin_vel_w[:,env._upright_rear_ids,:2]
    return upright(env)*torch.sum(torch.sum(velocity**2,dim=-1)*contacts,dim=1)


def rear_airtime_reward(env):
    robot,sensor,contacts,moving=rear_gait_state(env)
    first=sensor.compute_first_contact(env.step_dt)[:,env._upright_sensor_ids]
    duration=sensor.data.last_air_time[:,env._upright_sensor_ids]
    valid=torch.clamp(duration/.12,max=1.)*(duration>=.04)*first
    return upright(env)*moving*torch.sum(valid,dim=1)


def rear_single_support(env):
    robot,sensor,contacts,moving=rear_gait_state(env)
    return upright(env)*moving*(contacts.sum(dim=1)==1)


def rear_flight(env):
    robot,sensor,contacts,moving=rear_gait_state(env)
    return upright(env)*(contacts.sum(dim=1)==0)


def rear_clearance(env):
    robot,sensor,contacts,moving=rear_gait_state(env)
    height=robot.data.body_pos_w[:,env._upright_rear_ids,2]-env.scene.env_origins[:,2:3]
    return upright(env)*moving*torch.mean(torch.exp(-(height-.055)**2/.001)*(~contacts),dim=1)


def upright_nonfoot_contacts(env):
    sensor=env.scene['contact_forces']
    if not hasattr(env,'_upright_nonfoot_ids'):
        env._upright_nonfoot_ids=[i for i,n in enumerate(sensor.body_names) if not n.endswith('_foot')]
    force=torch.linalg.vector_norm(sensor.data.net_forces_w[:,env._upright_nonfoot_ids],dim=-1)
    return upright(env)*(force>1.).sum(dim=1)


def enable_stepping_rewards(cfg):
    """Correct the sliding optimum; actual foot lifts and quiet stance contacts are required."""
    cfg.rewards.velocity.weight=6.
    cfg.rewards.velocity.params={'std2':.04}
    cfg.rewards.height.weight=3.
    cfg.rewards.height.params={'target':.48,'std2':.01}
    cfg.rewards.rear_support.weight=.5
    cfg.rewards.rear_slip=RewTerm(func=rear_stance_slip,weight=-10.)
    cfg.rewards.rear_airtime=RewTerm(func=rear_airtime_reward,weight=15.)
    cfg.rewards.single_support=RewTerm(func=rear_single_support,weight=2.)
    cfg.rewards.rear_flight=RewTerm(func=rear_flight,weight=-3.)
    cfg.rewards.rear_clearance=RewTerm(func=rear_clearance,weight=1.5)
    cfg.rewards.nonfoot_contact=RewTerm(func=upright_nonfoot_contacts,weight=-3.)
