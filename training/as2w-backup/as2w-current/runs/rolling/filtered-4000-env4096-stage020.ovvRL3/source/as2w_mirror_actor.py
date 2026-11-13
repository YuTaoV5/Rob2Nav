"""Left/right reflection of the 60-feature AS2-W actor contract."""
import numpy as np


SWAP = np.arange(16) ^ 1
JOINT_SIGNS = np.r_[-np.ones(4), np.ones(12)].astype(np.float32)


def reflect_action(action):
    return action[SWAP] * JOINT_SIGNS


def reflect_observation(obs, flip_clock=False):
    reflected = obs.copy()
    reflected[[1, 2, 3, 5, 7]] *= -1.
    for start in (10, 26, 42):
        reflected[start:start + 16] = reflect_action(obs[start:start + 16])
    if flip_clock:
        reflected[58:60] *= -1.
    return reflected


def mirrored_actor(actor, flip_clock=False):
    def infer(obs):
        reflected = reflect_observation(obs, flip_clock)
        return .5 * (actor(obs) + reflect_action(actor(reflected)))

    return infer
