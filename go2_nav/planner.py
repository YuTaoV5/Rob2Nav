"""Reactive velocity rollout controller. No grid, SLAM map, or obstacle ground truth."""
import numpy as np


class ReactivePlanner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.turn_sign = 1.

    def act(self, points_body, goal_body, previous_command):
        c = self.cfg
        points = np.asarray(points_body)[:, :2]
        heading = np.arctan2(goal_body[1], goal_body[0])
        radius = c["robot_radius"] + .10
        # Pick a visible local corridor first. Scoring only progress toward the
        # final goal can make a safe zero command beat every necessary turn.
        steering_goal = np.asarray(goal_body, dtype=float)
        corridor_score = np.inf
        for angle in np.linspace(-np.pi, np.pi, 73):
            reach = min(2.2, np.linalg.norm(goal_body))
            endpoint = reach * np.array([np.cos(angle), np.sin(angle)])
            samples = np.linspace(.1, 1., 16)[:, None] * endpoint
            clearance = float(np.min(np.linalg.norm(points[:, None]-samples[None], axis=-1))) if len(points) else 8.
            if clearance < radius:
                continue
            score = np.linalg.norm(goal_body-endpoint) + .12*abs(angle) + .08/max(clearance, .1)
            if score < corridor_score:
                corridor_score, steering_goal = score, endpoint
        best_score, best = -np.inf, np.zeros(3)
        for vx in (0., .2, .4, c["max_vx"]):
            for wz in np.linspace(-c["max_wz"], c["max_wz"], 13):
                t = np.linspace(.15, 2., 12)
                if abs(wz) < 1e-6:
                    xy = np.stack((vx*t, np.zeros_like(t)), -1)
                else:
                    xy = np.stack((vx/wz*np.sin(wz*t), vx/wz*(1-np.cos(wz*t))), -1)
                clearance = float(np.min(np.linalg.norm(points[:, None]-xy[None], axis=-1))) if len(points) else 8.
                if clearance < radius:
                    continue
                delta = steering_goal - xy[-1]
                progress = np.linalg.norm(steering_goal) - np.linalg.norm(delta)
                desired = np.arctan2(delta[1], delta[0])
                angle_error = abs(np.arctan2(np.sin(desired-wz*2), np.cos(desired-wz*2)))
                score = 3*progress - .9*angle_error + .15*min(clearance, 1.5) - .07*abs(wz-previous_command[2])
                if score > best_score:
                    best_score, best = score, np.array([vx, 0., wz])
        if not np.isfinite(best_score):
            # Turn in place when no translation is feasible; do not use scene geometry.
            best = np.array([0., 0., self.turn_sign*c["max_wz"]])
        elif abs(best[2]) > .1:
            self.turn_sign = np.sign(best[2])
        if np.linalg.norm(goal_body) < c["goal_tolerance"]:
            best[:] = 0
        return best / np.array([c["max_vx"], c["max_vy"], c["max_wz"]])
