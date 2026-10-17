"""Joint-space jump primitives with bounded torques, layered on RoboGauge."""
import numpy as np

DEFAULT_JUMP = dict(crouch_height=.22, crouch_time=.18, extend_height=.40,
                    extend_time=.14, pushback=.10, front_lead=.0,
                    flight_time=.20, kp=45., kd=.8)


def leg_pose(height, foot_x=0.):
    length = .213
    radius = min(np.hypot(height, foot_x), 2*length-.005)
    knee = -2*np.arccos(radius/(2*length))
    hip = -knee/2 - np.arctan2(foot_x, height)
    return hip, max(knee, -2.70)


class JumpSkill:
    def __init__(self, parameters=None):
        self.p = dict(DEFAULT_JUMP, **(parameters or {}))
        self.start = None
        self.last_end = -100.

    def trigger(self, now):
        if self.start is None and now-self.last_end > 1.:
            self.start = now
            return True
        return False

    @property
    def active(self):
        return self.start is not None

    def targets(self, now, nominal, kp, kd, allow_release=True):
        if self.start is None:
            return nominal, kp, kd
        t = now-self.start
        p = self.p
        end = p["crouch_time"]+p["extend_time"]+p["flight_time"]+.24
        if t >= end and (allow_release or t >= end+.5):
            self.start = None
            self.last_end = now
            return nominal, kp, kd
        pose = nominal.copy().reshape(4, 3)
        if t < p["crouch_time"]:
            h, x = p["crouch_height"], 0.
        elif t < p["crouch_time"]+p["extend_time"]:
            h, x = p["extend_height"], -p["pushback"]
        elif t < p["crouch_time"]+p["extend_time"]+p["flight_time"]:
            h, x = .24, 0.
        else:
            h, x = .31, 0.
        for leg in range(4):
            lh = h
            if leg >= 2 and p["crouch_time"] <= t < p["crouch_time"]+p["front_lead"]:
                lh = p["crouch_height"]
            pose[leg, 0] = .1 if leg in (0, 2) else -.1
            pose[leg, 1:] = leg_pose(lh, x)
        damping = 2. if t >= p["crouch_time"]+p["extend_time"]+p["flight_time"] else p["kd"]
        return pose.ravel(), np.full(12, p["kp"]), np.full(12, damping)
