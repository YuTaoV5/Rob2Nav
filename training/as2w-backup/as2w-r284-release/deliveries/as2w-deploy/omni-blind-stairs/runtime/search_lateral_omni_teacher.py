"""Search a privileged lateral teacher with velocity and periodic yaw feedback."""

import argparse
from functools import lru_cache, partial
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

import evaluate_omni_stairs as evaluation
from search_flat_omni_teacher import LATERAL_CASES, StepTeacher


BOUNDS = (
    (.42, .82), (.18, .30), (.05, .16), (1.0, 4.5),
    (.3, 1.7), (.1, 1.2), (.55, .95), (-.15, .15),
    (-.10, .24), (-.22, .22), (-.22, .22), (.35, 1.5),
    (-.22, .22), (-.22, .22),
)
INITIAL = (.62, .25, .11, 2.3, .65, .7, .75, 0., .10, 0., 0., .9, 0., 0.)
SIDE = np.array((-1., 1., -1., 1.))
REFLECTION = np.array((1., -1., -1.))
MIRROR_JOINTS = np.arange(16) ^ 1


@lru_cache(maxsize=1)
def course(results):
    evaluation.sim.ROOT = results
    contract = json.loads((results / "fall-baseline/fall-recovery-contract.json").read_text())
    return contract, evaluation.sim.load_target_limits(contract), StepTeacher(contract)


class LateralFeedbackTeacher:
    def __init__(self, contract, geometry, params):
        self.dt = contract["control_dt"]
        self.geometry = geometry
        self.params = params
        self.integral = np.zeros(3)
        self.base = None
        self.samples = []
        self.side_sign = None

    def act(self, model, data, obs, target, tick, advance):
        if self.base is None:
            self.base = model.body("base_link").id
            self.side_sign = -1. if target[1] < 0. else 1.
        rotation = data.xmat[self.base].reshape(3, 3)
        body_velocity = rotation.T @ data.qvel[:3]
        error = target - np.r_[body_velocity[:2], data.qvel[5]]
        mirrored = self.side_sign < 0.
        if mirrored:
            target = target * REFLECTION
            error = error * REFLECTION
        self.integral = (self.integral + self.dt * error).clip(-.50, .50)
        if np.max(np.abs(target)) < .01:
            action = np.r_[np.zeros(12), (.65 * error[0] + .35 * SIDE * error[2]) / (.086 * 16.)]
        else:
            action = self.move(rotation[2, 1] * self.side_sign, target, tick, error)
        if mirrored:
            action = action[MIRROR_JOINTS]
            action[:4] *= -1.
        self.samples.append((obs.copy(), action.copy()))
        return action

    def move(self, gravity_y, target, tick, error):
        (period, duty, height, stride_gain, yaw_gain, shift_gain, rear_phase, start_phase,
         bias, yaw_sin, yaw_cos, gain, yaw_sin2, yaw_cos2) = self.params
        vx, vy, yaw = target
        cycle = tick * self.dt / period + start_phase
        phase = (cycle + np.array((0., .5, rear_phase, (rear_phase + .5) % 1.))) % 1.
        swinging = phase < duty
        fraction = (phase / duty).clip(0., 1.)
        pulse = np.where(swinging, np.sin(np.pi * fraction), 0.)
        path = np.where(swinging, fraction - .5, .5 - (phase - duty) / (1. - duty))
        lateral = vy + np.clip(.2 * error[1] + gain * self.integral[1], -.19, .19)
        stride = (lateral + yaw * self.geometry.foot_xy[:, 0]) * period * (1. - duty) * stride_gain
        vertical = height * pulse - .15 * gravity_y * SIDE
        delta = np.column_stack((np.zeros(4), np.clip(path * stride, -.18, .18), vertical))
        support = 1. - pulse
        centroid = support @ self.geometry.foot_xy / support.sum()
        delta[:, :2] -= shift_gain * (centroid - self.geometry.foot_xy.mean(axis=0))
        joint = np.einsum("lij,lj->il", self.geometry.inverse, delta).reshape(12)
        forward = np.clip(vx + bias + gain * (error[0] + 1.25 * self.integral[0]), -.8, .8)
        angle = 2. * np.pi * cycle
        feedforward = (yaw_sin * np.sin(angle) + yaw_cos * np.cos(angle)
                       + yaw_sin2 * np.sin(2. * angle) + yaw_cos2 * np.cos(2. * angle))
        differential = np.clip(.25 * yaw + yaw_gain * error[2] + .55 * self.integral[2] + feedforward, -.8, .8)
        wheel = (forward + SIDE * differential) / (.086 * 16.)
        return np.r_[joint.clip(-.75, .75) / .35, wheel].clip(-6., 6.)


def trial(params, results, command, seed):
    contract, limits, geometry = course(results)
    teacher = LateralFeedbackTeacher(contract, geometry, params)
    target = np.asarray(evaluation.FLAT_COMMANDS[command])
    row = evaluation.flat_episode(None, contract, limits, target, seed, evaluation.gait_clock, teacher.act)
    return dict(row, command=command, seed=seed), teacher.samples


def objective(params, results, train):
    rows = [trial(params, results, command, seed)[0] for command, seed in train]
    errors = np.asarray([
        (r["xy_mae_m_s"] / .15, r["yaw_mae_rad_s"] / .22,
         r["stop_speed_m_s"] / .12, r["stop_yaw_rate_rad_s"] / .15) for r in rows
    ])
    cost = errors.max(axis=1) + .35 * errors[:, :2].sum(axis=1) + .12 * errors[:, 2:].sum(axis=1)
    cost += 4. * np.asarray([row["fallen"] for row in rows])
    return float(.75 * cost.mean() + .25 * cost.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--generations", type=int, default=18)
    parser.add_argument("--start-parameters", type=Path)
    parser.add_argument("--group", choices=("all", "lateral", "diagonal"), default="all")
    args = parser.parse_args()
    results = args.results.resolve()
    commands = {
        "all": LATERAL_CASES,
        "lateral": LATERAL_CASES[:2],
        "diagonal": LATERAL_CASES[2:],
    }[args.group]
    train = [(name, 88371 + 100 * LATERAL_CASES.index(name) + replica)
             for name in commands for replica in range(2)]
    fitness = partial(objective, results=results, train=train)
    initial = INITIAL
    if args.start_parameters:
        initial = tuple(json.loads(args.start_parameters.read_text())["params"])
        if len(initial) == len(BOUNDS) - 2:
            initial += (0., 0.)

    def progress(intermediate_result):
        args.output.with_suffix(".progress.json").write_text(
            json.dumps({"params": intermediate_result.x.tolist(), "cost": intermediate_result.fun}) + "\n")
        print(f"generation={intermediate_result.nit} cost={intermediate_result.fun:.4f}", flush=True)

    result = differential_evolution(fitness, BOUNDS, x0=initial, popsize=4, seed=483,
                                    workers=args.workers, updating="deferred", polish=False,
                                    maxiter=args.generations, callback=progress)
    rows, records = [], []
    for command in commands:
        start = 88801 + 100 * LATERAL_CASES.index(command)
        for seed in range(start, start + 8):
            row, samples = trial(result.x, results, command, seed)
            rows.append(row)
            if row["passed"]:
                observations, actions = map(np.stack, zip(*samples))
                records.append((observations, actions, np.full(len(observations), seed)))
            print(json.dumps(row), flush=True)
    if records:
        obs, actions, seeds = map(np.concatenate, zip(*records))
        np.savez_compressed(args.output.with_suffix(".npz"), observations=obs, actions=actions, seed=seeds)
    args.output.write_text(json.dumps({"params": result.x.tolist(), "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
