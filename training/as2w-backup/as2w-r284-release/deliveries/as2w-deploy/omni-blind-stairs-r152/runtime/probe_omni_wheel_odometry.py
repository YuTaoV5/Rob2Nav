"""Evaluate a blind AS2-W velocity estimate from stance-wheel kinematics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_omni_stairs as evaluation
from search_lateral_omni_teacher import LateralFeedbackTeacher, course
from train_omni_lateral_history import CASES, gait, load_actor, reflect_action


HIP_X = np.array((.2119, .2119, -.2119, -.2119))
SIDE = np.array((1., -1., 1., -1.))
WHEEL_RADIUS = .086
LEG = .212


def wheel_profiles(model, joint_names):
    profiles = []
    for name in joint_names[12:]:
        body = model.body(name).id
        geom, = np.flatnonzero(model.geom_bodyid == body)
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
        orientation = np.empty(9)
        evaluation.sim.mj.mju_quat2Mat(orientation, model.geom_quat[geom])
        vertices = vertices @ orientation.reshape(3, 3).T + model.geom_pos[geom]
        axial = np.round(vertices[:, 1], 5)
        radius = np.linalg.norm(vertices[:, [0, 2]], axis=1)
        offsets = np.unique(axial)
        profiles.append((offsets, np.asarray([radius[axial == value].max() for value in offsets])))
    return profiles


def wheel_kinematics(angles, rates, wheel_rates, angular, up, profiles=()):
    hip, thigh, calf = angles
    dhip, dthigh, dcalf = rates
    combined = thigh + calf
    dsag_x = -LEG * (np.cos(thigh) * dthigh + np.cos(combined) * (dthigh + dcalf))
    dsag_z = LEG * (np.sin(thigh) * dthigh + np.sin(combined) * (dthigh + dcalf))
    sag_x = -LEG * (np.sin(thigh) + np.sin(combined))
    sag_z = -LEG * (np.cos(thigh) + np.cos(combined))
    sin, cos = np.sin(hip), np.cos(hip)
    offset = .1054 * SIDE
    hip_y = offset * cos - sag_z * sin
    hip_z = offset * sin + sag_z * cos
    center = np.column_stack((HIP_X + sag_x, .060 * SIDE + hip_y, hip_z))
    joint_velocity = np.column_stack((dsag_x, -hip_z * dhip - sin * dsag_z,
                                     hip_y * dhip + cos * dsag_z))
    axis = np.column_stack((np.zeros(4), cos, sin))
    axial = (axis @ up).clip(-1., 1.)
    normal = up - axial[:, None] * axis
    normal /= np.linalg.norm(normal, axis=1, keepdims=True).clip(min=.2)
    radial = -WHEEL_RADIUS * normal
    if profiles:
        for i, (offsets, radii) in enumerate(profiles):
            point = (offsets * axial[i] - radii * np.sqrt(1. - axial[i] ** 2)).argmin()
            radial[i] = offsets[point] * axis[i] - radii[point] * normal[i]
    wheel_angular = (angular + np.column_stack((dhip, np.zeros(4), np.zeros(4)))
                     + (dthigh + dcalf + wheel_rates)[:, None] * axis)
    contact = center + radial
    no_slip_velocity = -(joint_velocity + np.cross(angular, center) + np.cross(wheel_angular, radial))
    return no_slip_velocity, contact @ up


class WheelOdometry:
    def __init__(self, defaults, alpha=.35, profiles=()):
        self.defaults = np.asarray(defaults[:12]).reshape(3, 4)
        self.velocity = np.zeros(2)
        self.alpha = alpha
        self.profiles = profiles

    def update(self, obs):
        up = -np.asarray(obs[6:9])
        up /= max(np.linalg.norm(up), .2)
        angles = self.defaults + obs[10:22].reshape(3, 4)
        rates = obs[26:38].reshape(3, 4)
        wheel_velocity, height = wheel_kinematics(angles, rates, obs[38:42], obs[3:6], up, self.profiles)
        estimates = wheel_velocity[:, :2].clip(-2., 2.)
        support = np.exp(-((height - height.min()) / .026) ** 2)
        center = np.average(estimates, axis=0, weights=support)
        robust = support / np.maximum(1., np.linalg.norm(estimates - center, axis=1) / .22)
        estimate = np.average(estimates, axis=0, weights=robust)
        self.velocity += self.alpha * (estimate - self.velocity)
        return self.velocity.copy()


def trial(contract, limits, geometry, params, base, command, seed, use_odometry):
    target = np.asarray(evaluation.FLAT_COMMANDS[command])
    odometry = WheelOdometry(contract["default_joint_pos"])
    teacher = LateralFeedbackTeacher(contract, geometry, gait(target, params))
    fallback = evaluation.prior_flat_planner(base, contract["control_dt"])
    records = []
    base_id = None

    def plan(model, data, obs, cmd, tick, advance):
        nonlocal base_id
        if base_id is None:
            base_id = model.body("base_link").id
        estimate = odometry.update(obs)
        if abs(cmd[1]) < .01:
            return fallback(model, data, obs, cmd, tick, advance)
        rotation = data.xmat[base_id].reshape(3, 3)
        truth = (rotation.T @ data.qvel[:3])[:2]
        records.append((truth, estimate, [WHEEL_RADIUS * np.mean(obs[38:42]), 0.]))
        if not use_odometry:
            return teacher.act(model, data, obs, cmd, tick, advance)
        reflection = np.array((1., -1., -1.)) if cmd[1] < 0. else np.ones(3)
        error = (cmd - np.r_[estimate, obs[5]]) * reflection
        teacher.integral = (teacher.integral + contract["control_dt"] * error).clip(-.50, .50)
        action = teacher.move(-obs[7] * reflection[1], cmd * reflection, tick, error)
        return reflect_action(action) if cmd[1] < 0. else action

    row = evaluation.flat_episode(None, contract, limits, target, seed, evaluation.gait_clock, plan)
    window = np.asarray(records)[round(1.5 / contract["control_dt"]):]
    return dict(row, command=command, seed=seed, onboard_feedback=use_odometry,
                odometry_mae=np.abs(window[:, 1] - window[:, 0]).mean(axis=0).tolist(),
                encoder_only_mae=np.abs(window[:, 2] - window[:, 0]).mean(axis=0).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("anchor", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--teacher-parameters", type=Path, nargs=2, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    contract, limits, geometry = course(args.results.resolve())
    _, base = load_actor(args.anchor)
    params = np.asarray([json.loads(path.read_text())["params"] for path in args.teacher_parameters])
    rows = [
        trial(contract, limits, geometry, params, base, command, seed, use_odometry)
        for use_odometry in (False, True)
        for command in CASES
        for seed in range(103401, 103407)
    ]
    args.out.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    for enabled in (False, True):
        group = [row for row in rows if row["onboard_feedback"] == enabled]
        print(json.dumps({"onboard_feedback": enabled, "passed": sum(row["passed"] for row in group),
                          "trials": len(group), "odometry_mae": np.mean([r["odometry_mae"] for r in group], axis=0).tolist(),
                          "encoder_only_mae": np.mean([r["encoder_only_mae"] for r in group], axis=0).tolist()}))


if __name__ == "__main__":
    main()
