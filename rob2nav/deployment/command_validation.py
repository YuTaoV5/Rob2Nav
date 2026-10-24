"""Read-only validation of issued servo targets, separate from measured velocity."""
import numpy as np


class CommandRateValidation:
    def __init__(self, names, velocity_limits):
        self.names = tuple(names)
        self.limits = np.asarray(velocity_limits, dtype=float)
        if self.limits.shape != (len(self.names),) or not np.isfinite(self.limits).all() or np.any(self.limits <= 0):
            raise ValueError('Invalid per-joint velocity limits')
        self.previous = None
        self.samples = 0
        self.intervals = 0
        self.invalid = 0
        self.maximum = 0.
        self.peak = None

    def sample(self, time_s, target):
        q = np.asarray(target, dtype=float)
        self.samples += 1
        if q.shape != self.limits.shape or not np.isfinite(q).all() or not np.isfinite(time_s):
            self.invalid += 1
            self.previous = None
            return
        if self.previous is not None:
            old_time, old_q = self.previous
            dt = time_s - old_time
            if dt <= 0:
                self.invalid += 1
            else:
                ratios = np.abs(q - old_q) / dt / self.limits
                j = int(np.argmax(ratios))
                if ratios[j] > self.maximum:
                    self.maximum = float(ratios[j])
                    self.peak = dict(time_s=float(time_s), joint=self.names[j], interval_s=float(dt),
                                     delta_rad=float(q[j] - old_q[j]))
                self.intervals += 1
        self.previous = (float(time_s), q.copy())

    def report(self, expected_samples):
        covered = self.samples == expected_samples and self.samples >= 2 and self.intervals == self.samples - 1 and self.invalid == 0
        return dict(scope='all consecutive issued servo targets including transitions; first target has no preceding command',
                    samples=self.samples, intervals=self.intervals, expected_samples=expected_samples,
                    invalid_samples=self.invalid, coverage_complete=bool(covered),
                    max_command_rate_ratio=self.maximum, peak=self.peak,
                    passed=bool(covered and self.maximum <= 1.000001),
                    limitation='Does not certify raw policy continuity, acceleration, jerk or visual absence of twitching')
