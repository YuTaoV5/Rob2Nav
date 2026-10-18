"""Optional goal braking experiment; does not modify historical V4 rewards."""
import torch


def desired_velocity(delta):
    distance=delta.norm(dim=-1,keepdim=True)
    # Smoothly stop at the goal instead of paying for forward motion there.
    speed=(distance/.8).clamp(max=.6)
    return delta/distance.clamp(min=1e-6)*speed


def velocity_reward(env):
    robot=env.scene['robot']
    delta=env.scene.env_origins[:,:2]+torch.tensor([3.6,0.],device=env.device)-robot.data.root_pos_w[:,:2]
    return torch.exp(-(robot.data.root_lin_vel_w[:,:2]-desired_velocity(delta)).square().sum(-1)/.16)


def progress_reward(env):
    robot=env.scene['robot']
    delta=env.scene.env_origins[:,:2]+torch.tensor([3.6,0.],device=env.device)-robot.data.root_pos_w[:,:2]
    distance=delta.norm(dim=-1)
    projected=(robot.data.root_lin_vel_w[:,:2]*delta).sum(-1)/distance.clamp(min=1e-6)
    return projected.clamp(-1.,1.2)*(distance/.8).clamp(max=1.)


def configure(cfg):
    from isaaclab.managers import RewardTermCfg
    cfg.rewards.forward_progress=RewardTermCfg(func=progress_reward,weight=5.)
    cfg.rewards.goal_motion=RewardTermCfg(func=velocity_reward,weight=5.)
