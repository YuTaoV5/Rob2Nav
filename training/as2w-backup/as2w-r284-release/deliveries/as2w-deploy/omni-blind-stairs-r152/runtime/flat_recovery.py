"""Sensor-based trim for mixed lateral and yaw commands."""
from collections import deque

import numpy as np
from torch import nn

from train_recovery_student import numpy_actor

WIDTH, ACTIONS = 100, 6
HIDDEN = (96, 64)
SIDE = np.array((-1., 1., -1., 1.))
AXLE = np.array((1., 1., -1., -1.))


def adjustment_model():
    return nn.Sequential(nn.Linear(WIDTH, HIDDEN[0]), nn.ELU(),
                         nn.Linear(HIDDEN[0], HIDDEN[1]), nn.ELU(),
                         nn.Linear(HIDDEN[1], ACTIONS))


class FlatRecovery:
    def __init__(self, actor, geometry, state=None):
        self.actor = actor
        self.inverse = geometry.inverse
        self.history = deque(maxlen=7)
        self.training_action = None
        self.predict = None
        if state is not None:
            model = adjustment_model()
            model.load_state_dict(state)
            self.predict = numpy_actor(model)

    def features(self, obs):
        frame = np.r_[obs[3:9], obs[38:42] * .15]
        if self.history:
            self.history.append(frame)
        else:
            self.history.extend([frame] * 7)
        return np.r_[obs, *(self.history[i] for i in (0, 2, 4, 6))].astype(np.float32)

    def plan(self, model, data, obs, command, tick, advance):
        action = self.actor.plan(model, data, obs, command, tick, advance)
        features = self.features(obs) if self.predict is not None else None
        strength = np.clip((abs(command[1]) - .045) / .10, 0., 1.) * np.clip((abs(command[2]) - .045) / .12, 0., 1.)
        if strength == 0.:
            return action
        trim = self.predict(features) if self.predict is not None else self.training_action
        if trim is None or not np.any(trim):
            return action
        trim = np.clip(trim, -1., 1.)
        foot = np.zeros((4, 3))
        foot[:, 0] = .028 * trim[5] * SIDE
        foot[:, 1] = .026 * trim[2]
        foot[:, 2] = .024 * trim[3] * SIDE + .024 * trim[4] * AXLE
        legs = np.einsum("lij,lj->il", self.inverse, foot).reshape(12) / .35
        wheels = .23 * trim[0] + .24 * trim[1] * SIDE
        return (action + strength * np.r_[legs, wheels]).clip(-6., 6.)
