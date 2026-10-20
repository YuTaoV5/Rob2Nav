"""Identical velocity interface for live and batch MuJoCo navigation."""
import numpy as np


def action(env, planner, policy=None):
    if policy is not None:
        result, _ = policy.predict(env._observation(), deterministic=True)
        return result
    yaw = np.arctan2(env.rotation[1, 0], env.rotation[0, 0])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    velocity = planner.act(env.points, rotation.T@(env.goal-env.data.qpos[:2]), env.command)
    return velocity/np.array([env.cfg['max_vx'], env.cfg['max_vy'], env.cfg['max_wz']])
