"""Lateral stride trim and yaw-rate damping from the existing proprioceptive estimate."""
import numpy as np
from search_lateral_omni_teacher import SIDE


class FlatTrim:
    def __init__(self, actor, dt, weights):
        self.actor = actor
        self.move = actor.teacher.move
        self.weights = weights
        self.alpha = 1. - np.exp(-dt / weights[4])
        self.filtered = np.zeros(3)

    def __call__(self, gravity_y, target, tick, error):
        lateral, reverse_extra, x_gain, yaw_damping, _ = self.weights
        if self.actor.history.ticks == 1:
            self.filtered[:] = error
        else:
            self.filtered += self.alpha * (error - self.filtered)
        weight = np.clip((target[1] - .065) / .11, 0., 1.)
        reverse = np.clip((-target[0] - .08) / .25, 0., 1.)
        stride_target = target.copy()
        stride_target[1] += weight * np.clip(lateral + reverse_extra * reverse + .08 * self.filtered[1], -.08, .11)
        adjusted = error.copy()
        adjusted[2] += weight * yaw_damping * (self.filtered[2] - error[2])
        action = self.move(gravity_y, stride_target, tick, adjusted)
        action[12:] += weight * x_gain * self.filtered[0] / (.086 * 16.)
        return action.clip(-6., 6.)
