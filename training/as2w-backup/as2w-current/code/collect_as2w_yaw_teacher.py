"""Fit a proprioceptive wheel-leg teacher against actual joystick test speeds."""
import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
import torch

from eval_as2w_low_speed_course import run_flat
from search_flat_omni_teacher import StepTeacher
from train_stair_low_speed_course import OmniEnvironment
import mujoco_screen as sim


TRAIN_COMMANDS = ((0., .10, 0.), (0., -.10, 0.), (.071, .071, .20), (-.071, -.071, -.20))
DEV_COMMANDS = (*TRAIN_COMMANDS, (0., 0., .30), (0., 0., -.30))


def evaluate(params, assets, seed, commands):
    env = OmniEnvironment(assets, 1, stop_training=True)
    env.apply_motion_prior = False
    teacher = StepTeacher(env.contract)
    return [run_flat(env, teacher.policy(params), seed, command) for command in commands]


def score(params, assets):
    rows = evaluate(params, assets, 54101, TRAIN_COMMANDS)
    return float(np.mean([
        (r["tracking_m_s"] if r["tracking_m_s"] is not None else .5) / .10
        + .25 * (r["yaw_error_rad_s"] if r["yaw_error_rad_s"] is not None else 1.) / .20
        + 2. * r["terminated"] + .4 * (not r["held"])
        + .3 * r["hind_p95_rad"] + .03 * r["contact_frames"]
        for r in rows
    ]))


def collect_yaw(assets, params, output, archive):
    env = OmniEnvironment(assets, 1, stop_training=True)
    env.apply_motion_prior = False
    teacher = StepTeacher(env.contract)
    rows, accepted = [], []
    for index, seed in enumerate(range(56001, 56025)):
        for direction in (-1., 1.):
            command = (0., 0., direction * .30)
            actions, observations = [], []
            teacher_action = teacher.policy(params)

            def act(obs):
                action = teacher_action(obs).astype(np.float32)
                observations.append(obs[:58].copy())
                actions.append(action)
                return action

            row = run_flat(env, act, seed, command)
            row["accepted"] = bool(not row["terminated"] and row["held"] and not row["contact_frames"]
                                   and row["tracking_m_s"] is not None and row["tracking_m_s"] <= .06
                                   and row["yaw_error_rad_s"] is not None and row["yaw_error_rad_s"] <= .15)
            rows.append(row)
            if row["accepted"]:
                episode_id = 57001 + 2 * index + int(direction > 0)
                accepted.append((np.asarray(observations), np.asarray(actions), np.full(len(actions), episode_id)))
    output.write_text(json.dumps(rows, indent=2, default=lambda value: value.item()) + "\n")
    if accepted:
        obs, actions, seeds = (np.concatenate(values) for values in zip(*accepted))
        np.savez_compressed(archive, observations=obs, actions=actions, seed=seeds)
    print("accepted yaw demonstrations", len(accepted), "of", len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--collect-yaw", type=Path, help="Archive safe yaw rollouts on new seeds")
    parser.add_argument("--params-from", type=Path, help="Read prior search result for collection")
    args = parser.parse_args()
    torch.set_num_threads(1)
    sim.ROOT = args.assets.resolve()
    if args.collect_yaw:
        if not args.params_from:
            parser.error("--collect-yaw requires --params-from")
        params = json.loads(args.params_from.read_text())["params"]
        collect_yaw(args.assets, params, args.output, args.collect_yaw)
        return
    bounds = ((.48, .80), (.16, .42), (.025, .12), (-3., 4.),
              (.05, .65), (0., .65), (0., .45), (0., 1.4), (0., .035))

    def report(intermediate_result):
        print(f"generation={intermediate_result.nit} cost={intermediate_result.fun:.5f}", flush=True)

    optimum = differential_evolution(partial(score, assets=args.assets), bounds, seed=259,
                                    workers=args.workers, maxiter=args.generations,
                                    popsize=4, polish=False, updating="deferred", callback=report)
    baseline = (.62, .25, .11, 1.8, .62, .65, .15, .65, .01)
    rows = {}
    for name, params in (("existing_lateral", baseline), ("optimized", optimum.x)):
        rows[name] = [r for seed in range(54201, 54205) for r in evaluate(params, args.assets, seed, DEV_COMMANDS)]
    args.output.write_text(json.dumps({"params": optimum.x.tolist(), "cost": optimum.fun, "heldout": rows},
                                     indent=2, default=lambda value: value.item()) + "\n")


if __name__ == "__main__":
    main()
