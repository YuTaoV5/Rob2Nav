"""Learned low-dimensional corrections driven by wheel slip and gravity."""
import numpy as np

AXLE = np.array((1., 1., -1., -1.))
SIDE = np.array((1., -1., 1., -1.))


class StallResponse:
    def __init__(self, base, weights):
        self.base = base
        self.weights = weights
        self.stall = np.zeros(4)
        self.previous = None

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, obs):
        action = self.base(obs)
        if self.base.switch_tick is None or self.base.settling:
            self.stall.fill(0.)
            self.previous = None
            return action
        elapsed = self.base.route.tick - self.base.switch_tick
        if elapsed <= 8:
            return action
        drive, redistribute, lift, roll, damping, clearance = self.weights
        speed = .086 * obs[38:42]
        self.stall += .28 * (np.clip((.32 - speed) / .35, 0., 1.) - self.stall)
        strength = np.clip((elapsed - 8) / 8., 0., 1.)
        spread = self.stall - self.stall.mean()
        load = np.clip(-obs[7] * SIDE, -.25, .25)
        lift_m = (lift * spread + roll * load + clearance * AXLE * self.stall.mean()).clip(-.035, .035)
        result = action.copy()
        result[4:8] += strength * (3.2877543 / .35) * lift_m
        result[8:12] -= strength * (6.5755085 / .35) * lift_m
        result[12:] += strength * (drive * self.stall.mean() - redistribute * spread)
        if self.previous is not None:
            result[:12] += damping * strength * (self.previous - result[:12])
        self.previous = result[:12].copy()
        return result.clip(-6., 6.)
