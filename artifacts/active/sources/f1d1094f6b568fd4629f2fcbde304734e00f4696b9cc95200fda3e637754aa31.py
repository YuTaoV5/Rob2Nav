"""Finite local waypoint memory derived from measured coloured returns."""
import numpy as np
from .planner import ReactivePlanner


class TerrainNavigator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.local = ReactivePlanner(dict(cfg))
        self.waypoints = []
        self.water_handled = False

    def act(self, points, water_returns, position, yaw, goal, previous):
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        if not self.water_handled and len(water_returns) > 15:
            measured = water_returns[:, :2] @ rotation.T + position
            lo, hi = np.min(measured, axis=0), np.max(measured, axis=0)
            if hi[0]-lo[0] > .3 and hi[1]-lo[1] > .3:
                # Select a locally observed side and keep waypoints while looking away.
                # The locomotion prior has a low-speed deadband. Reach a broad
                # waypoint tolerance outside the inflated hazard, then turn.
                margin = self.cfg["robot_radius"]+.81
                left, right = hi[1]+margin, lo[1]-margin
                side = left if abs(left-position[1]) <= abs(right-position[1]) else right
                self.waypoints = [np.array([lo[0]-.75, side]), np.array([hi[0]+.75, side])]
                if self.cfg.get("water_recenter", False):
                    self.waypoints.append(np.array([hi[0]+2., goal[1]]))
                self.water_handled = True
        while self.waypoints and np.linalg.norm(self.waypoints[0]-position) < .55:
            self.waypoints.pop(0)
        target = self.waypoints[0] if self.waypoints else goal
        self.local.cfg["goal_tolerance"] = .20 if self.waypoints else self.cfg["goal_tolerance"]
        body_target = rotation.T @ (target-position)
        return self.local.act(points, body_target, previous)
