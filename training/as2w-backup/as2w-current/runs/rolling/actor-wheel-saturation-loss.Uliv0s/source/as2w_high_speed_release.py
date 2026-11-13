"""Separate high-speed braking from the subsequent ten-second hold."""
import math

import numpy as np


DECELERATION_M_S2 = 1.2
HOLD_S = 10.0


class HighSpeedRelease:
    def __init__(self, env):
        self.dt = env.dt
        self.origin = env.data.qpos[:2].copy()
        self.initial_speed = float(np.linalg.norm(env.data.qvel[:2]))
        self.max_brake_time = 1.0 + self.initial_speed / DECELERATION_M_S2
        self.max_brake_distance = (.05 + .25 * self.initial_speed
                                   + self.initial_speed**2 / (2.0 * DECELERATION_M_S2))
        self.max_frames = math.ceil((self.max_brake_time + HOLD_S) / self.dt)
        self.steady_frames = math.ceil(.5 / self.dt)
        self.hold_frames = math.ceil(HOLD_S / self.dt)
        self.max_distance = 0.0
        self.steady = 0
        self.stop_time = None
        self.hold_position = None
        self.hold_yaw = None
        self.hold_displacement = 0.0
        self.hold_heading = 0.0
        self.hold_speeds = []

    def sample(self, env, tick, rotation):
        position = env.data.qpos[:2]
        speed = float(np.linalg.norm(env.data.qvel[:2]))
        yaw_rate = abs(float(env.data.qvel[5]))
        heading = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        if self.stop_time is None:
            self.max_distance = max(self.max_distance, float(np.linalg.norm(position - self.origin)))
            self.steady = self.steady + 1 if speed <= .05 and yaw_rate <= .12 else 0
            if self.steady >= self.steady_frames:
                self.stop_time = (tick + 1) * self.dt
                self.hold_position = position.copy()
                self.hold_yaw = heading
            return False
        self.hold_speeds.append(speed)
        yaw_delta = (heading - self.hold_yaw + math.pi) % (2 * math.pi) - math.pi
        self.hold_heading = max(self.hold_heading, abs(math.degrees(yaw_delta)))
        self.hold_displacement = max(self.hold_displacement,
                                     float(np.linalg.norm(position - self.hold_position)))
        return len(self.hold_speeds) >= self.hold_frames

    def result(self, contacts, terminated):
        residual = float(np.quantile(self.hold_speeds, .95)) if self.hold_speeds else None
        entry_ok = self.initial_speed >= 1.70
        brake_ok = (self.stop_time is not None and self.stop_time <= self.max_brake_time
                    and self.max_distance <= self.max_brake_distance)
        hold_ok = (len(self.hold_speeds) >= self.hold_frames and self.hold_displacement <= .05
                   and self.hold_heading <= 5. and residual is not None and residual <= .035)
        return dict(passed=bool(entry_ok and brake_ok and hold_ok and not contacts and not terminated),
                    entry_ok=entry_ok, brake_ok=brake_ok, hold_ok=hold_ok,
                    entry_speed_m_s=self.initial_speed, stop_time_s=self.stop_time,
                    brake_limit_s=self.max_brake_time, brake_distance_m=self.max_distance,
                    brake_limit_m=self.max_brake_distance, hold_displacement_m=self.hold_displacement,
                    hold_yaw_drift_deg=self.hold_heading, hold_residual_p95_m_s=residual,
                    hold_duration_s=len(self.hold_speeds) * self.dt)
