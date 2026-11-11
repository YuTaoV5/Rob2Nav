"""Command and IMU gait prior for the AS2-W locomotion actor."""

import math

import torch
from torch import nn


GAIT_PERIOD = .62
STEERING = (.4536317028734116, .04637072561801772, .4049755449811857,
            .49481723150694257, .7612104631960321, .019069863003787935, 0., 0.)
LATERAL = (.25, .11, 1.8, .62, .65, .15, .65, .01)


class OmniMotionPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("offsets", torch.tensor((0., .5, .75, .25)))
        self.register_buffer("sides", torch.tensor((-1., 1., -1., 1.)))
        self.register_buffer("foot_xy", torch.tensor((
            (.2119, .1654), (.2119, -.1654), (-.2119, .1654), (-.2119, -.1654),
        )))
        # Nominal wheel-center Jacobian pseudoinverses, in FL, FR, RL, RR order.
        inverse_left = torch.tensor((
            (0., 3.3851986, 0.),
            (-3.3851986, -1.1730705, 3.2877543),
            (0., 2.346141, -6.5755085),
        ))
        inverse_right = torch.tensor((
            (0., 3.3851986, 0.),
            (-3.3851986, 1.1730705, 3.2877543),
            (0., -2.346141, -6.5755085),
        ))
        self.register_buffer("inverse", torch.stack((
            inverse_left, inverse_right, inverse_left, inverse_right,
        )))

    def _gait(self, phase, vx, vy, yaw, gyro_yaw, gravity_y, params):
        duty, height, stride_gain, yaw_gain, yaw_feedback, roll_feedback, shift_gain, stance_shift = params
        swing = phase < duty
        fraction = (phase / duty).clamp(0., 1.)
        lift = torch.where(swing, torch.sin(math.pi * fraction), 0.)
        intensity = ((vy.abs() + .22 * yaw.abs()) / .12).clamp(0., 1.)
        stride = (vy + yaw * self.foot_xy[:, 0]) * GAIT_PERIOD * (1. - duty) * stride_gain
        foot_y = torch.where(swing, -.5 + fraction, .5 - (phase - duty) / (1. - duty)) * stride
        foot_z = height * lift * intensity + (roll_feedback * gravity_y * self.sides).clamp(-.04, .04)
        support = 1. - lift
        centroid = support @ self.foot_xy / support.sum(dim=1, keepdim=True)
        body_shift = (centroid - self.foot_xy.mean(dim=0)) * shift_gain * intensity
        delta = torch.stack((
            -body_shift[:, :1].expand_as(foot_y),
            foot_y.clamp(-.13, .13) - body_shift[:, 1:2],
            foot_z - stance_shift * intensity,
        ), dim=-1)
        joints = torch.einsum("lij,nlj->nil", self.inverse, delta).flatten(1)
        wheel = (vx + self.sides * (yaw_gain * yaw + yaw_feedback * (yaw - gyro_yaw))) / (.086 * 16.)
        return torch.cat((joints.clamp(-.65, .65) / .35, wheel), dim=1).clamp(-6., 6.)

    def forward(self, action, command, gyro_yaw, gravity_y, elapsed):
        vx, vy, yaw = command.split(1, dim=1)
        phase = (elapsed[:, None] / GAIT_PERIOD + self.offsets).remainder(1.)
        steer = self._gait(phase, vx, vy, yaw, gyro_yaw, gravity_y, STEERING)
        lateral = self._gait(phase, vx, vy, yaw, gyro_yaw, gravity_y, LATERAL)
        mix = vy.abs() / (vy.abs() + .22 * yaw.abs()).clamp_min(.001)
        strength = torch.maximum(vy.abs() / .15, yaw.abs() / .45).clamp_max(1.)
        prior = torch.lerp(steer, lateral, mix)
        residual = (.1 + .3 * mix) * action
        return torch.lerp(action, prior + residual, strength).clamp(-6., 6.)
