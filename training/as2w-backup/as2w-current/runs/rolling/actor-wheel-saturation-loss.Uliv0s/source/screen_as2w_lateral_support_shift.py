"""Evaluate a simulated lateral stepping teacher with world-frame support targets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from train_stair_low_speed_course import OmniEnvironment, sim


class LateralTeacher:
    def __init__(self, env, period, lift, lead, preload, support_shift=0.):
        self.env, self.period, self.lift, self.lead = env, period, lift, lead
        self.wheels = sorted(env.wheel_index, key=env.wheel_index.get)
        data = env.data
        self.origin = data.qpos[:3].copy()
        rotation = data.xmat[env.base].reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        self.heading = np.array(((np.cos(yaw), -np.sin(yaw), 0.),
                                 (np.sin(yaw), np.cos(yaw), 0.), (0., 0., 1.)))
        self.anchors = data.xpos[self.wheels].copy()
        self.neutral = (self.anchors - self.origin) @ self.heading
        self.swing_start = self.anchors.copy()
        self.swing_target = self.anchors.copy()
        self.was_swinging = np.zeros(4, dtype=bool)
        self.body_target = self.origin.copy()
        self.support_shift = support_shift
        self.body_shift = np.zeros(3)
        self.offsets = np.array((0., .5, .75, .25))
        self.jacobian = np.zeros((3, env.model.nv))
        # At rest, the nominal PD target and the loaded joint angles differ.
        self.support_bias = preload * (env.defaults[:12] - data.qpos[env.qa[:12]])

    def action(self, command, elapsed):
        env, data = self.env, self.env.data
        velocity = self.heading @ np.array((0., command, 0.))
        lateral = self.heading[:, 1]
        desired = float((self.body_target - self.origin) @ lateral) + command * env.dt
        actual = float((data.qpos[:3] - self.origin) @ lateral)
        self.body_target = self.origin + lateral * np.clip(desired, actual - self.lead, actual + self.lead)
        phase = (elapsed / self.period + self.offsets) % 1.
        swinging = (phase < .24) & ((abs(command) > .015) | self.was_swinging)
        shift_target = np.zeros(3)
        if self.support_shift and swinging.any():
            support_center = self.anchors[~swinging, :2].mean(axis=0)
            shift_target[:2] = self.support_shift * np.clip(support_center - self.body_target[:2], -.08, .08)
        self.body_shift += .18 * (shift_target - self.body_shift)
        support_target = self.body_target + self.body_shift
        action = np.zeros(16, dtype=np.float32)
        rotation = data.xmat[env.base].reshape(3, 3)
        for leg, wheel in enumerate(self.wheels):
            if swinging[leg] and not self.was_swinging[leg]:
                self.swing_start[leg] = data.xpos[wheel]
                self.swing_target[leg] = (self.body_target + self.heading @ self.neutral[leg]
                                          + velocity * self.period * .50)
                self.swing_target[leg, 2] = self.anchors[leg, 2]
            if self.was_swinging[leg] and not swinging[leg]:
                self.anchors[leg] = self.swing_target[leg]
            target = self.anchors[leg].copy()
            if swinging[leg]:
                u = phase[leg] / .24
                blend = u * u * (3. - 2. * u)
                target = (1. - blend) * self.swing_start[leg] + blend * self.swing_target[leg]
                target[2] += self.lift * np.sin(np.pi * u)
            actual_offset = rotation.T @ (data.xpos[wheel] - data.qpos[:3])
            desired_offset = self.heading.T @ (target - support_target)
            joints = np.array((leg, leg + 4, leg + 8))
            dofs = np.asarray(env.va)[joints]
            sim.mj.mj_jacBody(env.model, data, self.jacobian, None, wheel)
            jac = rotation.T @ self.jacobian[:, dofs]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + .0025 * np.eye(3), desired_offset - actual_offset)
            support_scale = 1. if not swinging[leg] else 1. - np.sin(np.pi * phase[leg] / .24)
            q = (data.qpos[np.asarray(env.qa)[joints]] + np.clip(dq, -.30, .30)
                 + support_scale * self.support_bias[joints])
            action[joints] = (q - env.defaults[joints]) / .35
        self.was_swinging = swinging
        return np.clip(action, -3., 3.)


def evaluate(case):
    assets, seed, sign, params = case
    torch.set_num_threads(1)
    env = OmniEnvironment(assets, 1, stop_training=True)
    env.apply_motion_prior = False
    env.reset(seed=seed, options={"scene_seed": seed, "target": (0., 0., 0.)})
    env.next_command_tick = 10**9
    signed, cross, residual = [], [], []
    contacts = 0
    contact_bodies = {}
    teacher = None
    for tick in range(300):
        if tick == 30:
            teacher = LateralTeacher(env, *params)
            env.target_goal = np.array((0., .10 * sign, 0.))
        if tick == 195:
            env.target_goal.fill(0.)
        a = np.zeros(16, dtype=np.float32) if teacher is None else teacher.action(env.command[1], (tick - 30) * env.dt)
        _, _, terminated, truncated, _ = env.step(a)
        rotation = env.data.xmat[env.base].reshape(3, 3)
        velocity = (rotation.T @ env.data.qvel[:3])[:2]
        if 65 <= tick < 195:
            signed.append(float(sign * velocity[1]))
            cross.append(abs(float(velocity[0])))
        if tick >= 235:
            residual.append(float(np.linalg.norm(velocity)))
        bodies = {int(env.model.geom_bodyid[c.geom[1] if c.geom[0] == env.ground else c.geom[0]])
                  for c in env.data.contact if env.ground in c.geom}
        nonwheel = bodies - set(env.wheel_index)
        contacts += bool(nonwheel)
        for body in nonwheel:
            name = env.model.body(body).name
            contact_bodies[name] = contact_bodies.get(name, 0) + 1
        if terminated or truncated:
            break
    return dict(seed=seed, direction=sign, params=params, signed_m_s=mean(signed) if signed else None,
                cross_m_s=mean(cross) if cross else None, terminated=terminated, nonwheel_frames=contacts,
                contact_bodies=contact_bodies,
                held=bool(len(residual) == 65 and np.quantile(residual, .95) <= .05))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--support-shift", action="store_true", help="Compare a body shift toward three supporting wheels")
    args = parser.parse_args()
    settings = [(period, lift, .035, preload) for period in (.70, .95)
                for lift in (.035, .055) for preload in (0., .8, 1.15)]
    if args.support_shift:
        settings = [(period, .04, .035, .8, shift) for period in (.78, .96) for shift in (0., .35, .65)]
    cases = [(args.assets, seed, direction, params)
             for params in settings for seed in (61001, 61002, 61101, 61102) for direction in (-1, 1)]
    with ProcessPoolExecutor(max_workers=6 if args.support_shift else 12) as pool:
        rows = list(pool.map(evaluate, cases))
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    for params in settings:
        group = [r for r in rows if r["params"] == params]
        print(params, [round(mean(r["signed_m_s"] or 0. for r in group if r["direction"] == sign), 3)
                       for sign in (-1, 1)], "held", sum(r["held"] for r in group),
              "nonwheel", sum(r["nonwheel_frames"] for r in group), "terminated",
              sum(r["terminated"] for r in group), flush=True)


if __name__ == "__main__":
    main()
