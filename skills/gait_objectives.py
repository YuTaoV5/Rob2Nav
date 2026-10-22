"""Dense posterior stepping and measured landing objectives, without state writes."""
import torch


def alternating_rear_targets(time,frequency=2.,duty=.65):
    phase=torch.remainder((time-2.)*frequency,1.)
    phase=torch.stack([phase,torch.remainder(phase+.5,1.)],dim=-1)
    stance=phase<duty
    swing=torch.clamp((phase-duty)/(1.-duty),0.,1.)
    clearance=.022+.065*torch.sin(torch.pi*swing)
    return stance,clearance


def rear_alternation(env):
    from skills.fixstand import phase_time
    from skills.upright_walk.task import upright
    time=phase_time(env);robot=env.scene['robot'];sensor=env.scene['contact_forces']
    ids=robot.find_bodies(['RR_foot','RL_foot'],preserve_order=True)[0]
    sensor_ids=sensor.find_bodies(['RR_foot','RL_foot'],preserve_order=True)[0]
    if len(ids)!=2 or len(sensor_ids)!=2:raise ValueError('Expected RR and RL feet')
    loaded=torch.linalg.vector_norm(sensor.data.net_forces_w[:,sensor_ids],dim=-1)>4.
    stance,height=alternating_rear_targets(time)
    actual=robot.data.body_pos_w[:,ids,2]-env.scene.env_origins[:,2:3]
    contact_match=(loaded==stance).float().mean(-1)
    clearance=torch.exp(-((actual-height)/.035)**2).mean(-1)
    return ((time>=2.)&(time<5.))*upright(env)*(2.*contact_match+3.*clearance)


def configure_rear_alternation(cfg):
    from isaaclab.managers import RewardTermCfg
    cfg.rewards.rear_support.weight=0.
    cfg.rewards.single_support.weight=0.
    cfg.rewards.rear_clearance.weight=0.
    cfg.rewards.rear_alternation=RewardTermCfg(func=rear_alternation,weight=6.)


def landing_constraints(env,lower,upper,velocity):
    from skills.fixstand import phase_time
    robot=env.scene['robot'];names=robot.joint_names
    lo=torch.tensor([lower[n] for n in names],device=robot.device)
    hi=torch.tensor([upper[n] for n in names],device=robot.device)
    speed=torch.tensor([velocity[n] for n in names],device=robot.device)
    excess=torch.maximum(torch.relu(lo-robot.data.joint_pos),torch.relu(robot.data.joint_pos-hi)).max(-1).values
    overspeed=torch.relu((torch.abs(robot.data.joint_vel)/speed).max(-1).values-.85)
    sensor=env.scene['contact_forces']
    ids=[i for i,n in enumerate(sensor.body_names) if 'foot' not in n.lower() and 'calf' not in n.lower()]
    contacts=(torch.linalg.vector_norm(sensor.data.net_forces_w[:,ids],dim=-1)>4.).float().sum(-1)
    time=phase_time(env)
    return ((time>=2.4)&(time<5.))*((excess/.04)**2+10.*overspeed**2+contacts)


def configure_sideflip_landing(cfg,source):
    from pathlib import Path
    from xml.etree import ElementTree as ET
    from isaaclab.managers import RewardTermCfg
    limits={j.attrib['name']:j.find('limit').attrib for j in ET.parse(Path(source)/'resources/robots/go2/urdf/go2_description.urdf').getroot().findall('joint') if j.find('limit') is not None}
    cfg.rewards.landing_constraints=RewardTermCfg(func=landing_constraints,weight=-8.,params={
        'lower':{n:float(d['lower']) for n,d in limits.items()},'upper':{n:float(d['upper']) for n,d in limits.items()},
        'velocity':{n:float(d['velocity']) for n,d in limits.items()}})
    cfg.rewards.fixstand_peak_pose.weight=40.
