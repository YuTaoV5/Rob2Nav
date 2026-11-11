"""Fit a proprioceptive steering release and validate the integrated policy."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_omni_stairs as evaluation
from screen_omni_estimated_lateral import EstimatedLateralController
from search_lateral_omni_teacher import course
from train_omni_lateral_history import MIRROR, reflect_action, screen
from train_omni_routed_residual import arrays, make_route, residual_model
from train_omni_velocity_estimator import estimator_model
from train_recovery_student import make_actor, numpy_actor


class SteeringReleaseController(EstimatedLateralController):
    def __init__(self, base, estimator, params, contract, geometry, odometry, release):
        super().__init__(base, estimator, params, contract, geometry, odometry)
        self.release, self.mirror = release[:2]
        self.steering_only = len(release) > 2 and release[2]
        self.stop_start = None
        self.previous_command = np.zeros(3)

    def plan(self, model, data, obs, command, tick, advance):
        if np.any(command):
            self.previous_command = command.copy()
            self.stop_start = None
        elif (self.release and abs(self.previous_command[1:]).max() > .01
              and (not self.steering_only or abs(self.previous_command[1]) < .01)):
            if self.stop_start is None:
                self.stop_start = tick
            fraction = min(1., (tick - self.stop_start + 1) * self.dt / self.release)
            command = self.previous_command * (.5 + .5 * np.cos(np.pi * fraction))
        mirrored = self.mirror and command[2] < -.01 and abs(command[1]) < .01
        if mirrored or not np.array_equal(obs[:3] + (.4, 0., 0.), command):
            obs = obs.copy()
            obs[:3] = command - (.4, 0., 0.)
        if mirrored:
            command = command * (1., -1., -1.)
            obs[[1, 2, 3, 5, 7]] *= -1.
            for start in (10, 26, 42):
                obs[start:start + 16] = obs[start + MIRROR]
                obs[start:start + 4] *= -1.
        action = super().plan(model, data, obs, command, tick, advance)
        return reflect_action(action) if mirrored else action


def load(root, checkpoint):
    torch.set_num_threads(1)
    policy = torch.load(checkpoint, weights_only=True, map_location="cpu")
    contract, limits, geometry = course(root)
    anchor, primary = (numpy_actor(make_actor(policy[key])) for key in ("anchor", "stair"))
    correction, residual = estimator_model(), residual_model()
    correction.load_state_dict(policy["estimator"])
    residual.load_state_dict(policy["stair_residual"])
    correction, residual = numpy_actor(correction), numpy_actor(residual)
    params = np.asarray(policy["parameters"])
    odometry = arrays(policy, "odometry_profiles")

    def base():
        return make_route(policy, anchor, primary, residual, contract)

    def controller(route, release):
        return SteeringReleaseController(route, correction, params, contract, geometry, odometry, release)

    return policy, contract, limits, base, controller


def test_candidate(task):
    root, checkpoint, release, seed = task
    _, contract, limits, base, controller = load(root, checkpoint)
    commands = np.array((
        (0., 0., .65), (0., 0., -.65), (.36, 0., .5), (.36, 0., -.5),
        (0., .30, 0.), (0., -.30, 0.), (.30, .23, -.30), (.30, -.23, .30),
        (-.23, 0., -.40), (.26, -.19, -.44),
    ))
    rows = []
    for index, target in enumerate(commands):
        for trial in range(3):
            actor = controller(base(), release)
            case_seed = seed + index * 3 + trial
            row = evaluation.flat_episode(None, contract, limits, target, case_seed,
                                          evaluation.gait_clock, actor.plan)
            rows.append(dict(row, target=target.tolist(), seed=case_seed))
    return rows


def rank(rows):
    passes = sum(row["passed"] for row in rows)
    falls = sum(row["fallen"] for row in rows)
    stop_error = sum(row["stop_speed_m_s"] + .5 * row["stop_yaw_rate_rad_s"] for row in rows)
    return passes, -falls, -stop_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--steering-only", action="store_true")
    args = parser.parse_args()
    root, checkpoint = args.results.resolve(), args.checkpoint.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = (0., False)
    candidates = [(period, mirror) for period in (0., .10, .20, .32, .46, .64)
                  for mirror in (False, True)]
    if args.steering_only:
        candidates = [baseline] + [(period, mirror, True) for period in (.08, .14, .20, .26)
                                   for mirror in (False, True)] + [(0., True, True)]
    first_seed = 162101 if args.steering_only else 153001
    with ProcessPoolExecutor(args.workers) as pool:
        tasks = [(root, checkpoint, config, first_seed) for config in candidates]
        training = {}
        for config, rows in zip(candidates, pool.map(test_candidate, tasks)):
            training[config] = rows
            print("fit", config, rank(rows), flush=True)
        best = max(candidates, key=lambda config: rank(training[config]))
        choices = list(dict.fromkeys((baseline, best)))
        tasks = [(root, checkpoint, config, first_seed + 1000) for config in choices]
        holdout = dict(zip(choices, pool.map(test_candidate, tasks)))
    print("holdout", [(config, rank(rows)) for config, rows in holdout.items()], flush=True)
    report = {
        "fit": [{"config": config, "rows": rows} for config, rows in training.items()],
        "holdout": [{"config": config, "rows": rows} for config, rows in holdout.items()],
    }
    (args.output / "calibration.json").write_text(json.dumps(report, indent=2))
    if rank(holdout[best]) <= rank(holdout[baseline]):
        return
    policy, contract, limits, base, controller = load(root, checkpoint)
    policy["steering_release"] = list(best)
    torch.save(policy, args.output / "policy.pt")
    for confirmation in (False, True):
        result = screen(base, policy["anchor"], None, np.asarray(policy["parameters"]),
                        contract, limits, lambda route: controller(route, best), confirmation)
        name = "confirmation" if confirmation else "fixed"
        (args.output / f"{name}.json").write_text(json.dumps(result, indent=2, default=lambda v: v.item()))
        print(name, {key: rank(rows)[:1] if key == "flat" else sum(row["passed"] for row in rows)
                     for key, rows in result.items()}, flush=True)


if __name__ == "__main__":
    main()
