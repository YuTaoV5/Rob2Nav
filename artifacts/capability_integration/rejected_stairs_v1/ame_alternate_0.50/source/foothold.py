"""Privileged training reward for descending feet above missing support."""
import torch
from .mdp import FEET


def risk_score(relative_z,vertical_velocity,support_fraction):
    # Ignore the airborne high swing and feet already far below the platform.
    low=(relative_z>-.05)&(relative_z<.16)&(vertical_velocity<-.05)
    return (low.float()*(1-support_fraction).clamp(0,1)).sum(-1)


def descending_risk(env):
    robot=env.scene['robot'];ids=[robot.body_names.index(n) for n in FEET]
    height=robot.data.body_pos_w[:,ids,2]-env.scene.env_origins[:,None,2]
    velocity=robot.data.body_lin_vel_w[:,ids,2]
    probes=torch.stack([env.scene['probe_'+n[:2]].data.ray_hits_w[...,2] for n in FEET],1)
    support=(torch.isfinite(probes)&(probes-env.scene.env_origins[:,None,2,None]>-.055)).float().mean(-1)
    return risk_score(height,velocity,support)


def configure(cfg):
    from isaaclab.managers import RewardTermCfg
    cfg.rewards.descending_foot_risk=RewardTermCfg(func=descending_risk,weight=-2.)
