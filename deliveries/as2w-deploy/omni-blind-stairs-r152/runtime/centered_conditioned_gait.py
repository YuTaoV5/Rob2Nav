"""Independent command-conditioned tuning for forward and reverse mixed gait."""

import numpy as np


class CenteredConditionedGait:
    def __init__(self, controller, reverse_coefficients, forward_coefficients, center_coefficients, yaw_window=(0., .01)):
        self.controller = controller
        self.nominal = controller.params.copy()
        self.reverse_coefficients = np.asarray(reverse_coefficients)
        self.forward_coefficients = np.asarray(forward_coefficients)
        self.yaw_window = yaw_window
        self.center_coefficients = np.asarray(center_coefficients)

    def plan(self, model, data, obs, command, tick, advance):
        gait = self.nominal.copy()
        lateral = min(abs(command[1]) / .35, 1.)
        yaw = min(abs(command[2]) / .8, 1.)
        intensity = max(lateral, yaw)
        reverse = np.clip((-command[0] - .08) / .18, 0., 1.)
        active = np.clip((abs(command[1]) - .05) / .12, 0., 1.)
        forward = np.clip((command[0] - .08) / .18, 0., 1.)
        low, high = self.yaw_window
        forward *= np.clip((abs(command[2]) - low) / (high - low), 0., 1.)
        signed = (self.reverse_coefficients * reverse + self.forward_coefficients * forward) * active
        center = np.clip((.23 - abs(command[0])) / .19, 0., 1.)
        center *= np.clip((.45 - abs(command[2])) / .35, 0., 1.) * active
        centered = self.center_coefficients * center
        a = signed + centered
        gait[:, 0] *= 1. + .17 * a[0] * intensity
        gait[:, 1] = (gait[:, 1] + .07 * a[1] * intensity).clip(.15, .35)
        gait[:, 2] *= 1. + .24 * a[2] * intensity
        gait[:, 3] *= 1. + .32 * a[3] * lateral
        gait[:, 4] *= 1. + .42 * (signed[4] * yaw + centered[4] * lateral)
        gait[:, 5] *= 1. + .42 * a[5] * intensity
        gait[:, 11] *= 1. + .32 * a[6] * intensity
        gait[:, [9, 10, 12, 13]] *= 1. + .46 * (signed[7] * yaw + centered[7] * lateral)
        self.controller.params = gait
        return self.controller.plan(model, data, obs, command, tick, advance)


