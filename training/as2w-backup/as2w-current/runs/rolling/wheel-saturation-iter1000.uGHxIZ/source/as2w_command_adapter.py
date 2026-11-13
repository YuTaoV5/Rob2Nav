"""Command-conditioned yaw and wheel-position hold using canonical actor observations."""
import numpy as np


class CommandAdapter:
    def __init__(self, wheel_hold_gain=3., enable_yaw=True):
        self.wheel_hold_gain = wheel_hold_gain
        self.enable_yaw = enable_yaw
        self.reset()

    def reset(self):
        self.hold_reference = None
        self.hold_legs = None

    def apply(self, obs, action):
        action = np.asarray(action, dtype=np.float32).copy()
        # Match the float32 subtraction used by the actor observation.
        command = obs[:3] + np.array((.4, 0., 0.), dtype=np.float32)
        if not np.any(command):
            if self.hold_reference is None:
                self.hold_reference = obs[22:26].copy()
                self.hold_legs = obs[42:54].copy()
            delta = (obs[22:26] - self.hold_reference + 2. * np.pi) % (4. * np.pi) - 2. * np.pi
            action[:12] = self.hold_legs
            action[12:] = np.clip(-self.wheel_hold_gain * delta, -2., 2.) / 16.
        else:
            self.reset()
            yaw = command[2]
            planar = np.linalg.norm(command[:2])
            strength = min(abs(yaw) / .18, 1.) * max(0., 1. - planar / .025)
            if self.enable_yaw and strength:
                turn = np.clip((.45 * yaw + .55 * (yaw - obs[5])) / (.086 * 16.), -.8, .8)
                action[:12] *= 1. - strength
                action[12:] = (1. - strength) * action[12:] + strength * turn * np.array((-1., 1., -1., 1.))
        return np.clip(action, -6., 6.)
