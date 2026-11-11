"""Proprioceptive stair recovery in foot displacement and wheel target coordinates."""
from collections import deque

import numpy as np
from torch import nn

from train_recovery_student import numpy_actor

WIDTH = 100
HIDDEN = (96, 64)
ACTIONS = 8
AXLE = np.array((0, 0, 1, 1))
SIDE = np.array((-1., 1., -1., 1.))


def recovery_model():
    return nn.Sequential(nn.Linear(WIDTH, HIDDEN[0]), nn.ELU(),
                         nn.Linear(HIDDEN[0], HIDDEN[1]), nn.ELU(),
                         nn.Linear(HIDDEN[1], ACTIONS))


class ContactRecovery:
    def __init__(self, base, geometry, state=None, guard=None):
        self.base = base
        self.inverse = geometry.inverse
        self.frames = deque(maxlen=7)
        self.predict = None
        self.guard = guard
        self.heading = 0.
        if state is not None:
            model = recovery_model()
            model.load_state_dict(state)
            self.predict = numpy_actor(model)

    def __getattr__(self, name):
        return getattr(self.base, name)

    def features(self, obs):
        frame = np.r_[obs[3:9], obs[38:42] * .15]
        if not self.frames:
            self.frames.extend([frame] * 7)
        else:
            self.frames.append(frame)
        return np.r_[obs, *(self.frames[i] for i in (0, 2, 4, 6))].astype(np.float32)

    def move(self, obs, command):
        action = self.base(obs)
        trim_gain = 1.
        if self.guard is not None:
            floor, threshold, fade_start, heading_p, heading_d = self.guard
            straight = obs[0] + .4 > .10 and abs(obs[1]) + abs(obs[2]) < .06
            up = -obs[6:9]
            yaw_rate = obs[3:6] @ up / max(np.linalg.norm(up), .2)
            self.heading = (self.heading + self.base.dt * yaw_rate + np.pi) % (2. * np.pi) - np.pi if straight else 0.
            if self.base.switch_tick is not None:
                speed = .086 * obs[38:42].reshape(2, 2).mean(axis=1).min()
                stall = np.clip((threshold - speed) / .22, 0., 1.)
                fade = 1. if fade_start <= 0. else np.clip((fade_start + .4 - self.base.distance) / .4, 0., 1.)
                trim_gain = (floor + (1. - floor) * stall) * fade
                if heading_p or heading_d:
                    correction = np.clip(-heading_p * self.heading - heading_d * yaw_rate, -.16, .16)
                    action = action.copy()
                    action[12:] += SIDE * correction
                    action = action.clip(-6., 6.)
        if command is None or not np.any(command) or self.base.switch_tick is None or self.base.settling or trim_gain <= 0.:
            return action
        age = self.base.route.tick - self.base.switch_tick
        blend = trim_gain * min(1., age / 5.)
        command = np.clip(command, -1., 1.)
        displacement = np.zeros((4, 3))
        displacement[:, 0] = .052 * command[2:4][AXLE]
        displacement[:, 2] = .064 * command[:2][AXLE] + .022 * SIDE * command[4]
        delta = np.einsum("lij,lj->il", self.inverse, displacement).reshape(12)
        wheel = .40 * command[5:7][AXLE] + .15 * SIDE * command[7]
        return (action + blend * np.r_[np.clip(delta / .35, -.95, .95), wheel]).clip(-6., 6.)

    def __call__(self, obs):
        features = self.features(obs)
        command = self.predict(features) if self.predict is not None else None
        return self.move(obs, command)
