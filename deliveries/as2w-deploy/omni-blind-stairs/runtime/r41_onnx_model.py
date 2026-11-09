"""Tensor-only R41 control step with explicit state for ONNX Runtime."""

import math

import numpy as np
import torch
from torch import nn

from omni_motion_prior import OmniMotionPrior
from train_omni_lateral_history import SCALE
from train_omni_routed_residual import residual_model
from train_omni_stair_contact_gate import FEATURES, contact_gate
from train_omni_velocity_estimator import estimator_model
from train_recovery_student import make_actor


# Clock, previous action, lateral/gate histories, odometry, route, gait, stop.
STATE_PARTS = (1, 16, 7 * 58, 7 * len(FEATURES), 2, 7, 7, 18)
STATE_SIZE = sum(STATE_PARTS)
OBS_SIZE = 41


def sine(value):
    return value.float().sin().double()


def cosine(value):
    return value.float().cos().double()


class R41Step(nn.Module):
    """One 20 ms tick. Observation holds command, gyro, gravity, q, and dq."""

    def __init__(self, policy, contract, geometry):
        super().__init__()
        self.anchor = make_actor(policy["anchor"])
        self.stair = make_actor(policy["stair"])
        self.residual = residual_model()
        self.residual.load_state_dict(policy["stair_residual"])
        self.estimator = estimator_model()
        self.estimator.load_state_dict(policy["estimator"])
        self.gate = contact_gate()
        gate = policy["contact_gate"]
        self.gate.load_state_dict(gate["gate_state_dict"])
        self.register_buffer("gate_center", gate["center"].float())
        self.register_buffer("gate_scale", gate["scale"].float())
        self.gate_threshold = float(gate["threshold"])
        self.gate_count = int(gate.get("required_consecutive", 3))
        self.gate_blend = float(gate.get("blend_seconds", .18))
        self.prior = OmniMotionPrior()
        self.dt = float(contract["control_dt"])
        settings = policy["stair_residual_config"]
        self.residual_scale = float(settings["scale"])
        self.stall_gated = bool(settings["stall_gated"])
        self.leg_seconds, self.brake, self.yaw_gain, self.level = map(float, policy["stop_stance"])
        if self.level != 0. or policy.get("steering_release"):
            raise ValueError("The exporter expects the validated R41 stop and steering configuration")
        buffers = {
            "defaults": contract["default_joint_pos"],
            "gait_table": policy["parameters"],
            "reverse": policy["reverse_gait_conditioning"],
            "sides": (-1., 1., -1., 1.),
            "hip_x": (.2119, .2119, -.2119, -.2119),
            "foot_xy": geometry.foot_xy,
            "inverse": geometry.inverse,
            "command_offset": (.4, 0., 0.),
            "zero_legs": np.zeros(12),
        }
        for name, value in buffers.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float64))
        order = np.arange(16) ^ 1
        mirror = np.arange(58)
        sign = np.ones(58, dtype=np.float32)
        sign[[1, 2, 3, 5, 7]] = -1
        for start in (10, 26, 42):
            mirror[start:start + 16] = start + order
            sign[start:start + 4] = -1
        self.register_buffer("mirror_obs", torch.from_numpy(mirror))
        self.register_buffer("mirror_sign", torch.from_numpy(sign))
        self.register_buffer("mirror_action", torch.from_numpy(order))
        self.register_buffer("action_sign", torch.tensor([-1.] * 4 + [1.] * 12, dtype=torch.float64))
        self.register_buffer("lateral_scale", torch.from_numpy(SCALE))
        self.register_buffer("history_indices", torch.tensor([0, 2, 4, 6]))
        self.register_buffer("gate_indices", torch.as_tensor(FEATURES))
        for name in ("odometry_profiles", "wheel_profiles"):
            profiles = policy.get(name, ())
            if len(profiles) not in (0, 4):
                raise ValueError(f"Invalid {name} wheel count")
            for i, (offset, radius) in enumerate(profiles):
                self.register_buffer(f"{name}_{i}", torch.stack((offset, radius)).double())
            setattr(self, f"has_{name}", bool(profiles))

    def wheel_geometry(self, q, dq, angular, up, name):
        hip, thigh, calf = q[:12].reshape(3, 4).unbind(0)
        dhip, dthigh, dcalf = dq[:12].reshape(3, 4).unbind(0)
        combined = thigh + calf
        combined_rate = dthigh + dcalf
        sag_x = -.212 * (sine(thigh) + sine(combined))
        sag_z = -.212 * (cosine(thigh) + cosine(combined))
        dsag_x = -.212 * (cosine(thigh) * dthigh + cosine(combined) * combined_rate)
        dsag_z = .212 * (sine(thigh) * dthigh + sine(combined) * combined_rate)
        sin, cos = sine(hip), cosine(hip)
        offset = -.1054 * self.sides
        hip_y, hip_z = offset * cos - sag_z * sin, offset * sin + sag_z * cos
        center = torch.stack((self.hip_x + sag_x, -.060 * self.sides + hip_y, hip_z), dim=1)
        joint_rate = torch.stack((dsag_x, -hip_z * dhip - sin * dsag_z, hip_y * dhip + cos * dsag_z), dim=1)
        zero = torch.zeros_like(hip)
        axis = torch.stack((zero, cos, sin), dim=1)
        axial = (axis @ up).clamp(-1., 1.)
        normal = up - axial[:, None] * axis
        normal = normal / torch.linalg.vector_norm(normal, dim=1, keepdim=True).clamp_min(.2)
        radial = -.086 * normal
        if getattr(self, f"has_{name}"):
            vectors = []
            for wheel in range(4):
                offset, radius = getattr(self, f"{name}_{wheel}").unbind(0)
                cost = offset * axial[wheel] - radius * (1. - axial[wheel].square()).clamp_min(0.).sqrt()
                idx = torch.argmin(cost, dim=0).reshape(1)
                shift = torch.index_select(offset, 0, idx)[0]
                scale = torch.index_select(radius, 0, idx)[0]
                vectors.append(shift * axis[wheel] - scale * normal[wheel])
            radial = torch.stack(vectors)
        omega = angular.expand_as(axis)
        wheel_omega = omega + torch.stack((dhip, zero, zero), dim=1) + (combined_rate + dq[12:])[:, None] * axis
        velocity = -(joint_rate + torch.linalg.cross(omega, center, dim=1)
                     + torch.linalg.cross(wheel_omega, radial, dim=1))
        return velocity, (center + radial) @ up

    def routed_base(self, obs, command, q, dq, up, tick, history, route, called):
        age, count, switch, distance, raised, planar, settling = route.unbind(0)
        straight = (command[0] > .10) & (command[1:3].abs().sum() < .06)
        frame = torch.index_select(obs.float(), 0, self.gate_indices)
        updated = torch.where(age == 0, frame.expand_as(history), torch.cat((history[1:], frame[None, :])))
        features = torch.index_select(updated, 0, self.history_indices).reshape(-1)
        probability = self.gate(((features - self.gate_center) / self.gate_scale).clamp(-8., 8.)).sigmoid()[0]
        count = torch.where(probability >= self.gate_threshold, count + 1., 0.)
        switch = torch.where((switch == 0) & (count >= self.gate_count), age + 1., switch)
        engaged = straight & (switch > 0.)
        blend = torch.where(engaged, ((age - switch + 2.) * self.dt / self.gate_blend).clamp(0., 1.), 0.)
        anchor = self.anchor(obs.float()).clamp(-6., 6.)
        primary = self.stair(obs.float()).clamp(-6., 6.)
        residual = self.residual(obs.float()).clamp(-.48, .48)
        speeds = torch.sort(dq[12:]).values
        weight = self.residual_scale
        if self.stall_gated:
            weight = weight * ((.28 - .086 * speeds[:2].mean()) / .22).clamp(0., 1.)
        stair = (primary + weight * residual).clamp(-6., 6.)
        action = (1. - blend) * anchor.double() + blend * stair.double()
        distance = torch.where(engaged, distance + self.dt * (.086 * speeds[1:3].mean()).clamp(0., .65), 0.)
        _, height = self.wheel_geometry(q, dq, obs[3:6], up, "wheel_profiles")
        spread = height.max() - height.min()
        raised = torch.where(engaged, raised + (spread > .067).double(), 0.)
        ready = engaged & (raised >= 4.) & (distance >= 1.15) & (spread <= .010) & (up[2] >= .978)
        planar = torch.where(ready, planar + 1., 0.)
        settling = torch.where(engaged & ((planar >= 5.) | (settling > 0.)), settling + 1., 0.)
        settle = (settling * self.dt / .16).clamp_max(1.)
        wheels = (command[0].clamp_max(.4) - .22 * self.sides * obs[5]) / (.086 * 16.)
        action = (1. - settle) * action + settle * torch.cat((self.zero_legs, wheels))
        next_route = torch.stack((age + 1., count, switch, distance, raised, planar, settling))
        next_route = torch.where(straight, next_route, torch.zeros_like(next_route))
        next_route = torch.where(called, next_route, route)
        updated = torch.where(called & straight, updated, history)
        driven = self.prior(action.float()[None, :], command.float()[None, :], obs[5:6].float()[None, :],
                            obs[7:8].float()[None, :], (tick * self.dt).float().reshape(1))[0]
        return driven.double(), updated, next_route

    def gait_parameters(self, command):
        lateral = (command[1].abs() / .35).clamp_max(1.)
        yaw = (command[2].abs() / .8).clamp_max(1.)
        intensity = torch.maximum(lateral, yaw)
        active = ((command[1].abs() - .05) / .12).clamp(0., 1.)
        reverse = ((-command[0] - .08) / .18).clamp(0., 1.)
        a = self.reverse * reverse * active
        blend = ((command[0].abs() - .06) / .25).clamp(0., 1.)
        positive = (1. - blend) * self.gait_table[0] + blend * self.gait_table[1]
        negative = (1. - blend) * self.gait_table[2] + blend * self.gait_table[3]
        p = torch.where(command[1] < 0., negative, positive)
        return torch.stack((
            p[0] * (1. + .17 * a[0] * intensity),
            (p[1] + .07 * a[1] * intensity).clamp(.15, .35),
            p[2] * (1. + .24 * a[2] * intensity),
            p[3] * (1. + .32 * a[3] * lateral),
            p[4] * (1. + .42 * a[4] * yaw),
            p[5] * (1. + .42 * a[5] * intensity),
            p[6], p[7], p[8],
            p[9] * (1. + .46 * a[7] * yaw),
            p[10] * (1. + .46 * a[7] * yaw),
            p[11] * (1. + .32 * a[6] * intensity),
            p[12] * (1. + .46 * a[7] * yaw),
            p[13] * (1. + .46 * a[7] * yaw),
        ))

    def lateral_action(self, obs, command, q, dq, up, history, odometry, lateral_state):
        old_side, ticks, old_period, offset = lateral_state[:4].unbind(0)
        integral = lateral_state[4:]
        active = command[1].abs() > .01
        side = torch.where(active, torch.sign(command[1]), 0.)
        same = (side == old_side) & active
        ticks = torch.where(same, ticks, 0.)
        offset = torch.where(same, offset, 0.)
        integral = torch.where(same, integral, torch.zeros_like(integral))
        params = self.gait_parameters(command)
        period, duty, height, stride_gain, yaw_gain, shift_gain, rear_phase, start_phase, bias, ys, yc, gain, ys2, yc2 = params.unbind(0)
        elapsed = ticks * self.dt
        period_key = period.float().double()
        change = elapsed * (old_period.clamp_min(.1).reciprocal() - period_key.reciprocal())
        offset = offset + torch.where(same & (old_period > 0.) & (old_period != period_key), change, 0.)
        phase = 2. * math.pi * (elapsed / period + offset)
        frames = torch.index_select(history, 0, self.history_indices)
        reflected = torch.index_select(frames, 1, self.mirror_obs) * self.mirror_sign
        frames = torch.where(side < 0., reflected, frames)
        encoder_velocity, wheel_height = self.wheel_geometry(q, dq, obs[3:6], up, "odometry_profiles")
        estimates = encoder_velocity[:, :2].clamp(-2., 2.)
        support = torch.exp(-((wheel_height - wheel_height.min()) / .026).square())
        center = (estimates * support[:, None]).sum(0) / support.sum()
        robust = support / (torch.linalg.vector_norm(estimates - center, dim=1) / .22).clamp_min(1.)
        velocity = (estimates * robust[:, None]).sum(0) / robust.sum()
        odometry = odometry + .35 * (velocity - odometry)
        sign = torch.where(side < 0., -1., 1.)
        reflected_command = torch.stack((command[0], command[1] * sign, command[2] * sign))
        encoder = torch.stack((odometry[0], odometry[1] * sign))
        features = torch.cat(((frames * self.lateral_scale).reshape(-1).double(),
                              torch.stack((sine(phase), cosine(phase), (elapsed / 4.).clamp_max(1.))),
                              encoder))
        estimated = encoder + self.estimator(features.float()).clamp(-.55, .55).double()
        error = reflected_command - torch.cat((estimated, (obs[5] * sign).reshape(1)))
        integral = (integral + self.dt * error).clamp(-.5, .5)
        cycle = elapsed / period + offset + start_phase
        shifts = torch.stack((torch.zeros_like(cycle), torch.full_like(cycle, .5),
                              rear_phase, torch.remainder(rear_phase + .5, 1.)))
        step = torch.remainder(cycle + shifts, 1.)
        fraction = (step / duty).clamp(0., 1.)
        swing = step < duty
        pulse = torch.where(swing, sine(math.pi * fraction), 0.)
        path = torch.where(swing, fraction - .5, .5 - (step - duty) / (1. - duty))
        vx, vy, yaw = reflected_command.unbind(0)
        lateral = vy + (.2 * error[1] + gain * integral[1]).clamp(-.19, .19)
        stride = (lateral + yaw * self.foot_xy[:, 0]) * period * (1. - duty) * stride_gain
        vertical = height * pulse + .15 * obs[7] * sign * self.sides
        support = 1. - pulse
        shift = shift_gain * ((support @ self.foot_xy) / support.sum() - self.foot_xy.mean(0))
        delta = torch.stack((torch.zeros_like(pulse) - shift[0],
                             (path * stride).clamp(-.18, .18) - shift[1], vertical), dim=1)
        joint = torch.einsum("lij,lj->il", self.inverse, delta).reshape(12)
        forward = (vx + bias + gain * (error[0] + 1.25 * integral[0])).clamp(-.8, .8)
        angle = 2. * math.pi * cycle
        feed = ys * sine(angle) + yc * cosine(angle) + ys2 * sine(2. * angle) + yc2 * cosine(2. * angle)
        yaw_drive = (.25 * yaw + yaw_gain * error[2] + .55 * integral[2] + feed).clamp(-.8, .8)
        wheel = (forward + self.sides * yaw_drive) / (.086 * 16.)
        action = torch.cat((joint.clamp(-.75, .75) / .35, wheel)).clamp(-6., 6.)
        action = torch.where(side < 0., torch.index_select(action, 0, self.mirror_action) * self.action_sign, action)
        next_lat = torch.cat((torch.stack((side, ticks + 1., period_key, offset)), integral))
        return action, odometry, torch.where(active, next_lat, torch.zeros_like(next_lat)), active

    def apply_stop(self, action, command, gyro, state):
        steer, ticks = state[:2].unbind(0)
        last = state[2:]
        moving = command.abs().max() > 1e-4
        steer = torch.where(moving, (command[1:3].abs().sum() > .04).double(), steer)
        stopping = (~moving) & (steer > 0.)
        ticks = torch.where(stopping, ticks + 1., 0.)
        last = torch.where(moving, action, last)
        fraction = (ticks * self.dt / self.leg_seconds).clamp_max(1.)
        blend = fraction.square() * (3. - 2. * fraction)
        wheel_blend = (ticks * self.dt / self.brake).clamp_max(1.)
        wheel = (1. - wheel_blend) * last[12:] - wheel_blend * self.yaw_gain * gyro * self.sides / 16.
        stop = torch.cat(((1. - blend) * last[:12], wheel)).clamp(-6., 6.)
        return torch.where(stopping, stop, action), torch.cat((torch.stack((steer, ticks)), last))

    def forward(self, observation, state):
        raw = observation.reshape(OBS_SIZE).double()
        tick, previous, history, gate_history, odometry, route, lateral, stop = torch.split(
            state.reshape(STATE_SIZE).double(), STATE_PARTS
        )
        tick = tick[0]
        command, angular, gravity = raw[:3], raw[3:6], raw[6:9]
        q, dq = raw[9:25], raw[25:41]
        wheels = torch.remainder(q[12:] + 2. * math.pi, 4. * math.pi) - 2. * math.pi
        positions = torch.cat((q[:12], wheels)) - self.defaults
        phase = tick * self.dt * (2. * math.pi / .62)
        obs = torch.cat((command - self.command_offset, angular, gravity, self.command_offset[:1],
                         positions, dq, previous, torch.stack((sine(phase), cosine(phase)))))
        up = -gravity / torch.linalg.vector_norm(gravity).clamp_min(.2)
        history = history.float().reshape(7, 58)
        frame = obs[:58].float()[None, :]
        history = torch.where(tick == 0., frame.expand_as(history), torch.cat((history[1:], frame)))
        base, gate_history, route = self.routed_base(obs, command, q, dq, up, tick,
                                                   gate_history.float().reshape(7, -1), route,
                                                   command[1].abs() < .14)
        gait, odometry, lateral, active = self.lateral_action(obs, command, q, dq, up, history, odometry, lateral)
        mix = (command[1].abs() / .14).clamp_max(1.)
        action = torch.where(active, (1. - mix) * base + mix * gait, base)
        action, stop = self.apply_stop(action, command, angular[2], stop)
        next_state = torch.cat(((tick + 1.).reshape(1), action, history.reshape(-1).double(),
                               gate_history.reshape(-1).double(), odometry, route, lateral, stop))
        return action.float().reshape(1, 16), next_state.float().reshape(1, STATE_SIZE)
