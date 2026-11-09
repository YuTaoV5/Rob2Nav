"""Estimate periodic yaw error from gyro feedback in the lateral gait."""
import numpy as np


class YawHarmonic:
    def __init__(self, profile, dt):
        self.gain, self.memory, self.delay, self.second = profile
        self.dt = dt
        self.coefficients = np.zeros(4)
        self.previous_cycle = None

    def reset(self):
        self.coefficients.fill(0.)
        self.previous_cycle = None

    def update(self, cycle, period, error, lateral):
        if self.previous_cycle is not None:
            step = cycle - self.previous_cycle
            if step <= 0. or abs(step - self.dt / period) > .045:
                self.reset()
        self.previous_cycle = cycle
        angle = 2. * np.pi * cycle
        delayed = angle - 2. * np.pi * self.delay / period
        basis = np.array((np.sin(angle), np.cos(angle), np.sin(2.*angle), np.cos(2.*angle)))
        history = np.array((np.sin(delayed), np.cos(delayed), np.sin(2.*delayed), np.cos(2.*delayed)))
        strength = np.clip((lateral - .055) / .105, 0., 1.)
        if strength == 0.:
            self.coefficients *= .88
            return 0.
        alpha = self.dt / (self.memory * period)
        correction = self.gain * np.dot(self.coefficients, basis * (1., 1., self.second, self.second))
        self.coefficients += alpha * (2. * np.clip(error, -1.1, 1.1) * history - self.coefficients)
        np.clip(self.coefficients, -.6, .6, out=self.coefficients)
        return float(strength * np.clip(correction, -.23, .23))
