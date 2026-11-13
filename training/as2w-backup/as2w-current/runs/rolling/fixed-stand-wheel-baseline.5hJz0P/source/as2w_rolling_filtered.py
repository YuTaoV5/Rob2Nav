"""Action filtering and phase costs for the AS2-W low-speed rolling profile."""
import math

import torch


def filter_leg_residual(previous, action, dt, cutoff_hz):
    """First-order causal filter in normalized actor action units."""
    alpha = -math.expm1(-2. * math.pi * cutoff_hz * dt)
    return torch.lerp(previous, action, alpha)


def rolling_phase_cost(env):
    return env.rolling_phase_penalty


def update_rolling_phase_cost(env, command):
    robot = env.scene["robot"].data
    xy = robot.root_pos_w.torch[:, :2]
    speed = robot.root_lin_vel_b.torch[:, :2].norm(dim=1)
    yaw = robot.root_ang_vel_b.torch[:, 2].abs()
    moving = (command[:, 0].abs() >= .008) | (command[:, 2].abs() >= .015)
    if not hasattr(env, "rolling_motion_seen"):
        env.rolling_motion_seen = torch.zeros_like(moving)
        env.rolling_quiet_time = torch.zeros_like(speed)
        env.rolling_release_xy = xy.clone()
        env.rolling_park_xy = xy.clone()
        env.rolling_park_latched = torch.zeros_like(moving)
    env.rolling_motion_seen |= moving
    parked = env.rolling_motion_seen & ~moving
    env.rolling_release_xy[moving | ~env.rolling_motion_seen] = xy[moving | ~env.rolling_motion_seen]
    quiet = parked & (speed < .025) & (yaw < .06)
    env.rolling_quiet_time[:] = torch.where(quiet, env.rolling_quiet_time + env.step_dt, 0.)
    settled = env.rolling_quiet_time >= .50
    env.rolling_park_latched &= parked
    latch = settled & ~env.rolling_park_latched
    env.rolling_park_xy[latch] = xy[latch]
    env.rolling_park_latched |= latch
    travel = (xy - env.rolling_release_xy).norm(dim=1)
    drift = (xy - env.rolling_park_xy).norm(dim=1)
    ids = env.as2w_joint_ids[:12]
    error = (robot.joint_pos.torch[:, ids] - robot.default_joint_pos.torch[:, ids]).abs()
    stand_error = ((error - .12).clamp_min(0.) / .12).mean(dim=1)
    braking = speed / .10 + yaw / .20
    braking_cost = .5 * braking + .25 * travel / .10
    parked_cost = braking + .4 * drift / .02 + stand_error
    return parked * torch.where(env.rolling_park_latched, parked_cost, braking_cost).clamp(max=5.)
