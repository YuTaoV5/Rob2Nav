"""Optional fixed training targets. Ideal rays score rewards, never actor inputs."""
import torch
from .mdp import FEET


def body_height_error(env,target):
    robot=env.scene['robot'];root=robot.data.root_pos_w
    hits=env.scene['height_scanner'].data.ray_hits_w
    valid=torch.isfinite(hits).all(-1)
    distance=(hits[...,:2]-root[:,None,:2]).square().sum(-1)
    index=distance.masked_fill(~valid,float('inf')).argmin(-1)
    ground=hits[torch.arange(root.shape[0],device=root.device),index,2]
    height=root[:,2]-ground
    active=torch.isfinite(height)&(height>0)&(height<.8)
    return torch.where(active,(height-target).square(),torch.zeros_like(height))


def swing_clearance_error(env,target):
    robot=env.scene['robot'];sensor=env.scene['contact_forces']
    ids=[robot.body_names.index(n) for n in FEET]
    contact_ids=[sensor.body_names.index(n) for n in FEET]
    hits=torch.stack([env.scene['probe_'+n[:2]].data.ray_hits_w[...,2] for n in FEET],1)
    ground=hits.masked_fill(~torch.isfinite(hits),-float('inf')).amax(-1)
    height=robot.data.body_pos_w[:,ids,2]-ground
    moving=robot.data.body_lin_vel_w[:,ids,:2].norm(dim=-1)>.1
    free=sensor.data.net_forces_w[:,contact_ids].norm(dim=-1)<5.
    active=torch.isfinite(height)&(height>.02)&(height<.5)&moving&free
    return torch.where(active,(height-target).square(),torch.zeros_like(height)).mean(-1)


def configure(cfg,targets):
    from isaaclab.managers import RewardTermCfg
    if 'body_height_target_m' in targets:
        cfg.rewards.body_height_error=RewardTermCfg(func=body_height_error,weight=-1.,params={'target':targets['body_height_target_m']})
    if 'swing_clearance_target_m' in targets:
        cfg.rewards.swing_clearance_error=RewardTermCfg(func=swing_clearance_error,weight=-.5,params={'target':targets['swing_clearance_target_m']})
