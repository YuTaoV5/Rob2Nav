"""Reward mapped setpoint continuity over the complete action, without filtering state."""
import math


def validate(weight, margin):
    if type(weight) not in (int,float) or not math.isfinite(weight) or not 0<=weight<=100:
        raise ValueError('Invalid target rate weight')
    if type(margin) not in (int,float) or not math.isfinite(margin) or not .1<=margin<=1.:
        raise ValueError('Invalid target rate margin')


def penalty(env, names, velocity, margin=.8):
    import torch
    term=env.action_manager.get_term('JointPositionAction')
    delta=term.target_step_delta
    speed=torch.tensor([velocity[n] for n in names],device=delta.device,dtype=delta.dtype)
    ratio=torch.abs(delta)/(float(env.step_dt)*speed)
    return torch.relu(ratio.max(-1).values-margin)**2


def configure(cfg, source, names, weight=0., margin=.8):
    validate(weight,margin)
    if not weight:return
    from pathlib import Path
    from xml.etree import ElementTree as ET
    from isaaclab.managers import RewardTermCfg
    urdf=ET.parse(Path(source)/'resources/robots/go2/urdf/go2_description.urdf').getroot()
    velocity={j.attrib['name']:float(j.find('limit').attrib['velocity'])
              for j in urdf.findall('joint') if j.find('limit') is not None}
    cfg.rewards.mapped_target_rate=RewardTermCfg(func=penalty,weight=-float(weight),
        params={'names':list(names),'velocity':velocity,'margin':float(margin)})
