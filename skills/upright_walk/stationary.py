"""Stationary rear-foot balance, distinct from the walking objective."""
def balance(env):
    import torch
    from skills.fixstand import phase_time
    from .task import upright
    robot=env.scene['robot'];sensor=env.scene['contact_forces']
    rear=sensor.find_bodies(['RR_foot','RL_foot'],preserve_order=True)[0]
    front=sensor.find_bodies(['FR_foot','FL_foot'],preserve_order=True)[0]
    loaded=torch.linalg.vector_norm(sensor.data.net_forces_w[:,rear],dim=-1)>4.
    clear=(torch.linalg.vector_norm(sensor.data.net_forces_w[:,front],dim=-1)<4.).all(-1)
    still=torch.exp(-torch.sum(robot.data.root_lin_vel_w[:,:2]**2,dim=-1)/.0064
                    -torch.sum(robot.data.root_ang_vel_b**2,dim=-1)/.25
                    -torch.mean(robot.data.joint_vel**2,dim=-1)/2.)
    t=phase_time(env)
    return ((t>=2.)&(t<5.))*upright(env)*loaded.all(-1)*clear*still


def configure(cfg,source):
    from pathlib import Path
    from xml.etree import ElementTree as ET
    from isaaclab.managers import RewardTermCfg
    from skills.fixstand import urdf_speed_penalty
    from .task import upright_nonfoot_contacts
    cfg.commands.base_velocity.ranges.lin_vel_x=(0.,0.)
    cfg.commands.base_velocity.ranges.lin_vel_y=(0.,0.)
    cfg.commands.base_velocity.ranges.ang_vel_z=(0.,0.)
    cfg.commands.base_velocity.rel_standing_envs=1.
    cfg.rewards.stationary_balance=RewardTermCfg(func=balance,weight=12.)
    cfg.rewards.rear_support.weight=4.
    cfg.rewards.height.params['parameters']={'target':.48,'std2':.01}
    cfg.rewards.nonfoot_contact=RewardTermCfg(func=upright_nonfoot_contacts,weight=-10.)
    limits={j.attrib['name']:float(j.find('limit').attrib['velocity']) for j in
            ET.parse(Path(source)/'resources/robots/go2/urdf/go2_description.urdf').getroot().findall('joint') if j.find('limit') is not None}
    cfg.rewards.stationary_speed_margin=RewardTermCfg(func=urdf_speed_penalty,weight=-20.,params={'limits':limits,'margin':.9})
    cfg.rewards.joint_limit.weight=-20.;cfg.rewards.action_rate.weight=-.03
    cfg.rewards.joint_acc.weight=-5e-7
