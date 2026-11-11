"""Stop an established gait using encoder targets and IMU feedback."""

import numpy as np


class StopStance:
    def __init__(self, base, dt, settings):
        self.base = base
        self.dt = dt
        self.leg_seconds, self.brake, self.yaw_gain, self.level = settings
        self.steering = False
        self.count = 0
        self.last = np.zeros(16)
        self.release = np.zeros(16)

    def plan(self, model, data, obs, command, tick, advance):
        base = self.base.plan(model, data, obs, command, tick, advance)
        if np.max(np.abs(command)) > 1e-4:
            self.steering = abs(command[1]) + abs(command[2]) > .04
            self.count = 0
            self.last = base.copy()
            return base
        if not self.steering or self.leg_seconds <= 0.:
            return base
        if not self.count:
            self.release = self.last.copy()
        self.count += 1
        u = min(1., self.count * self.dt / self.leg_seconds)
        mix = u * u * (3. - 2. * u)
        legs = self.release[:12] * (1. - mix)
        if self.level:
            side = np.array((-1., 1., -1., 1.))
            ends = np.array((1., 1., -1., -1.))
            height = np.clip(self.level * (obs[7] * side + obs[6] * ends), -.026, .026)
            legs[4:8] += mix * 3.2877543 * height / .35
            legs[8:12] -= mix * 6.5755085 * height / .35
        wheel_time = min(1., self.count * self.dt / self.brake)
        differential = self.yaw_gain * obs[5] * np.array((1., -1., 1., -1.))
        wheels = (1. - wheel_time) * self.release[12:] + wheel_time * differential / 16.
        return np.r_[legs, wheels].clip(-6., 6.)


