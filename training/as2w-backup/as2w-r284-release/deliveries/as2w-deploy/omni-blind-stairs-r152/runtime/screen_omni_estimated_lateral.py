"""Evaluate one integrated AS2-W policy with a proprioceptive lateral estimator."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_omni_stairs as evaluation
from probe_omni_wheel_odometry import WheelOdometry, wheel_kinematics, wheel_profiles
from screen_omni_contact_routing import ContactRoutedActor
from search_lateral_omni_teacher import LateralFeedbackTeacher, course
from train_omni_lateral_history import History, gait, load_actor, reflect_action, screen
from train_omni_velocity_estimator import estimator_model
from train_recovery_student import make_actor, numpy_actor


class CommandGuard:
    def __init__(self, gate, anchor, stair, dt):
        self.gate, self.anchor, self.stair, self.dt = gate, anchor, stair, dt
        self.route = None

    @property
    def switch_tick(self):
        return self.route.switch_tick if self.route is not None else None

    def __call__(self, obs):
        straight = obs[0] + .4 > .10 and abs(obs[1]) + abs(obs[2]) < .06
        if not straight:
            self.route = None
            return self.anchor(obs)
        if self.route is None:
            self.route = ContactRoutedActor(self.gate, self.anchor, self.stair, self.dt)
        return self.route(obs)


class StairLevelingActor(CommandGuard):
    def __init__(self, gate, anchor, stair, dt, defaults, gain, profiles=()):
        super().__init__(gate, anchor, stair, dt)
        self.defaults = np.asarray(defaults[:12]).reshape(3, 4)
        self.gain = gain
        self.profiles = profiles
        self.zero_rates = np.zeros((3, 4))
        self.zero_wheels = np.zeros(4)
        self.zero_angular = np.zeros(3)

    def heights(self, angles, up):
        return wheel_kinematics(angles, self.zero_rates, self.zero_wheels, self.zero_angular, up, self.profiles)[1]

    def __call__(self, obs):
        action = super().__call__(obs)
        if not self.gain or self.switch_tick is None or -obs[8] < .96:
            return action
        up = -obs[6:9]
        up /= max(np.linalg.norm(up), .2)
        current = self.heights(self.defaults + obs[10:22].reshape(3, 4), up)
        # Apply load balancing only when the wheels are already near a common plane.
        weight = np.clip((.032 - np.ptp(current)) / .018, 0., 1.)
        if weight == 0.:
            return action
        target = self.heights(self.defaults + .35 * action[:12].reshape(3, 4), up)
        higher = .75 * (target - np.median(target)) + .25 * (current - np.median(current))
        extension = weight * self.gain * higher.clip(-.024, .024)
        action = action.copy()
        action[4:8] -= 3.2877543 * extension / .35
        action[8:12] += 6.5755085 * extension / .35
        return action.clip(-6., 6.)


class StairPlaneActor(StairLevelingActor):
    def __init__(self, gate, anchor, stair, dt, defaults, profiles, minimum_distance=1.15):
        super().__init__(gate, anchor, stair, dt, defaults, 0., profiles)
        self.minimum_distance = minimum_distance
        self.distance = 0.
        self.raised = 0
        self.planar = 0
        self.settling = 0
        self.tick = 0
        self.settled_tick = None

    def __call__(self, obs):
        action = super().__call__(obs)
        self.tick += 1
        if self.switch_tick is None:
            self.distance = 0.
            self.raised = self.planar = self.settling = 0
            return action
        self.distance += self.dt * float(np.clip(.086 * np.median(obs[38:42]), 0., .65))
        up = -obs[6:9]
        up /= max(np.linalg.norm(up), .2)
        current = self.heights(self.defaults + obs[10:22].reshape(3, 4), up)
        spread = np.ptp(current)
        self.raised += spread > .067
        ready = self.raised >= 4 and self.distance >= self.minimum_distance and spread <= .010 and up[2] >= .978
        self.planar = self.planar + 1 if ready else 0
        if self.planar >= 5 or self.settling:
            if self.settled_tick is None:
                self.settled_tick = self.tick - 1
            self.settling += 1
        if not self.settling:
            return action
        sides = np.array((-1., 1., -1., 1.))
        drive = min(obs[0] + .4, .4)
        wheels = (drive - .22 * sides * obs[5]) / (.086 * 16.)
        target = np.r_[np.zeros(12), wheels]
        mix = min(1., self.settling * self.dt / .16)
        return (1. - mix) * action + mix * target


class EstimatedLateralController:
    def __init__(self, base, estimator, params, contract, geometry, odometry_profiles=()):
        self.dt = contract["control_dt"]
        self.base = evaluation.prior_flat_planner(base, self.dt)
        self.estimator, self.params = estimator, params
        self.odometry = WheelOdometry(contract["default_joint_pos"], profiles=odometry_profiles)
        self.history = History()
        self.teacher = LateralFeedbackTeacher(contract, geometry, params[0])
        self.side = 0

    def plan(self, model, data, obs, command, tick, advance):
        params = gait(command, self.params)
        velocity = self.odometry.update(obs)
        features, mirrored = self.history.observe(obs, command, self.dt, params[0])
        if features is None:
            self.side = 0
            self.teacher.integral.fill(0.)
            if getattr(self.teacher, "yaw_harmonic", None) is not None:
                self.teacher.yaw_harmonic.reset()
            return self.base(model, data, obs, command, tick, advance)
        side = -1. if mirrored else 1.
        if self.side != side:
            self.teacher.integral.fill(0.)
            if getattr(self.teacher, "yaw_harmonic", None) is not None:
                self.teacher.yaw_harmonic.reset()
            self.side = side
        reflection = np.array((1., side, side))
        encoder = velocity * reflection[:2]
        correction = self.estimator(np.r_[features, encoder]).clip(-.55, .55)
        estimated = encoder + correction
        error = command * reflection - np.r_[estimated, obs[5] * side]
        self.teacher.integral = (self.teacher.integral + self.dt * error).clip(-.50, .50)
        self.teacher.params = params
        action = self.teacher.move(-obs[7] * side, command * reflection, self.history.phase_tick, error)
        if mirrored:
            action = reflect_action(action)
        mix = min(1., abs(command[1]) / .14)
        if mix < 1.:
            action = (1. - mix) * self.base(model, data, obs, command, tick, advance) + mix * action
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("estimator", type=Path)
    parser.add_argument("stair", type=Path)
    parser.add_argument("gate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--export", type=Path)
    parser.add_argument("--leveling-gain", type=float, default=0.)
    parser.add_argument("--feedback-parameters", type=Path)
    parser.add_argument("--plane-settle", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    contract, limits, geometry = course(args.results.resolve())
    checkpoint = torch.load(args.estimator, weights_only=True, map_location="cpu")
    params = np.asarray(checkpoint["parameters"])
    odometry_profiles = [(a.numpy(), r.numpy()) for a, r in checkpoint.get("odometry_profiles", ())]
    if args.feedback_parameters:
        params = np.asarray(json.loads(args.feedback_parameters.read_text())["parameters"])
        if params.shape not in ((2, 14), (4, 14)):
            parser.error("Expected two shared or four direction-specific 14-parameter profiles")
    state = checkpoint["anchor"]
    anchor = numpy_actor(make_actor(state))
    stair_state, stair = load_actor(args.stair)
    gate = torch.load(args.gate, weights_only=True, map_location="cpu")
    correction = estimator_model()
    correction.load_state_dict(checkpoint["estimator"])
    estimator = numpy_actor(correction)
    dt = contract["control_dt"]
    profiles = wheel_profiles(evaluation.sim.load_flat_model(True, True), contract["joint_names"]) if (
        args.leveling_gain or args.plane_settle
    ) else ()

    def base_factory():
        if args.plane_settle:
            return StairPlaneActor(gate, anchor, stair, dt, contract["default_joint_pos"], profiles)
        return StairLevelingActor(gate, anchor, stair, dt, contract["default_joint_pos"], args.leveling_gain, profiles)

    def controller_factory(base):
        return EstimatedLateralController(base, estimator, params, contract, geometry, odometry_profiles)

    report = screen(base_factory, state, None, params, contract, limits,
                    controller_factory, args.confirmation)
    args.output.write_text(json.dumps(report, indent=2, default=lambda value: value.item()) + "\n")
    if args.export:
        torch.save({"anchor": state, "stair": stair_state, "contact_gate": gate,
                    "estimator": checkpoint["estimator"], "parameters": params.tolist(),
                    "odometry_profiles": checkpoint.get("odometry_profiles", ()),
                    "leveling_gain": args.leveling_gain,
                    "plane_settle": args.plane_settle,
                    "wheel_profiles": [(torch.as_tensor(a), torch.as_tensor(r)) for a, r in profiles]}, args.export)
    print(json.dumps({name: [sum(row["passed"] for row in rows), len(rows)] for name, rows in report.items()}))


if __name__ == "__main__":
    main()
