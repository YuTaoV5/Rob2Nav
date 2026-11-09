"""Adapt gait parameters smoothly in reverse diagonal motion."""

import numpy as np


class ReverseConditionedGait:
    def __init__(self, controller, coefficients):
        self.controller = controller
        self.nominal = controller.params.copy()
        self.coefficients = np.asarray(coefficients)

    def plan(self, model, data, obs, command, tick, advance):
        gait = self.nominal.copy()
        lateral = min(abs(command[1]) / .35, 1.)
        yaw = min(abs(command[2]) / .8, 1.)
        intensity = max(lateral, yaw)
        reverse = np.clip((-command[0] - .08) / .18, 0., 1.)
        active = np.clip((abs(command[1]) - .05) / .12, 0., 1.)
        a = self.coefficients * reverse * active
        gait[:, 0] *= 1. + .17 * a[0] * intensity
        gait[:, 1] = (gait[:, 1] + .07 * a[1] * intensity).clip(.15, .35)
        gait[:, 2] *= 1. + .24 * a[2] * intensity
        gait[:, 3] *= 1. + .32 * a[3] * lateral
        gait[:, 4] *= 1. + .42 * a[4] * yaw
        gait[:, 5] *= 1. + .42 * a[5] * intensity
        gait[:, 11] *= 1. + .32 * a[6] * intensity
        gait[:, [9, 10, 12, 13]] *= 1. + .46 * a[7] * yaw
        self.controller.params = gait
        return self.controller.plan(model, data, obs, command, tick, advance)


