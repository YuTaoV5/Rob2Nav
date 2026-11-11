"""Command-based longitudinal action reference for the blind AS2-W residual policy."""
import numpy as np


def longitudinal_prior(observation):
    obs = np.asarray(observation, dtype=np.float32)
    vx = obs[..., 0] + np.float32(.4)
    vy, yaw = obs[..., 1], obs[..., 2]
    weight = np.clip((vx - .15) / .10, 0., 1.)
    weight *= np.clip(1. - np.abs(vy) / .08 - np.abs(yaw) / .20, 0., 1.)
    prior = np.zeros((*obs.shape[:-1], 16), dtype=np.float32)
    hip = .30 * np.clip((vx - .20) / 1.80, 0., 1.) / .35
    prior[..., 2] = -hip
    prior[..., 3] = hip
    wheel_y = np.array((.15, -.15, .15, -.15), dtype=np.float32)
    turn = -.65 * obs[..., 5]
    prior[..., 12:] = np.clip((vx[..., None] - turn[..., None] * wheel_y) / .086, -30., 30.) / 16.
    return (prior * weight[..., None]).astype(np.float32, copy=False)
