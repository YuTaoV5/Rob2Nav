"""Search a flat steering and side-step teacher for the blind policy."""

import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

import evaluate_omni_stairs as evaluation


CASES = ("left", "right", "turn_left", "turn_right", "diagonal_left", "arc_right")
LATERAL_CASES = ("left", "right", "diagonal_left", "diagonal_right")
BOUNDS = ((.44, .96), (.20, .46), (.04, .13), (.4, 2.2),
          (.05, 1.3), (.0, .9), (.0, .9))


class StepTeacher:
    def __init__(self, contract):
        model = evaluation.sim.load_flat_model(True, True)
        data = evaluation.sim.mj.MjData(model)
        joints = [model.joint(name) for name in contract["joint_names"]]
        data.qpos[[j.qposadr[0] for j in joints]] = contract["default_joint_pos"]
        evaluation.sim.mj.mj_forward(model, data)
        feet = [model.body(name).id for name in contract["joint_names"][12:]]
        self.foot_xy = data.xpos[feet, :2].copy()
        jacobian = np.zeros((3, model.nv))
        inverse = []
        for leg, body in enumerate(feet):
            evaluation.sim.mj.mj_jacBody(model, data, jacobian, None, body)
            axes = [joints[leg + 4 * k].dofadr[0] for k in range(3)]
            inverse.append(np.linalg.pinv(jacobian[:, axes]))
        self.inverse = np.stack(inverse)

    def policy(self, params):
        period, duty, height, stride_gain, yaw_gain, yaw_feedback, roll_feedback, *shift = params
        tick = 0
        offsets = np.array([0., .5, .75, .25])
        sides = np.array([-1., 1., -1., 1.])

        def act(obs):
            nonlocal tick
            vx, vy, yaw = obs[:3] + (.4, 0., 0.)
            gyro_yaw = obs[5]
            phase = (tick * .02 / period + offsets) % 1.
            tick += 1
            swing = phase < duty
            fraction = np.clip(phase / duty, 0., 1.)
            intensity = np.clip((abs(vy) + .22 * abs(yaw)) / .12, 0., 1.)
            stride = (vy + yaw * self.foot_xy[:, 0]) * period * (1. - duty) * stride_gain
            foot_y = np.where(swing, -.5 + fraction, .5 - (phase - duty) / (1. - duty)) * stride
            foot_z = np.where(swing, height * np.sin(np.pi * fraction), 0.) * intensity
            foot_z += np.clip(roll_feedback * obs[7] * sides, -.04, .04)
            delta = np.column_stack((np.zeros(4), np.clip(foot_y, -.13, .13), foot_z))
            if shift:
                # Keep the projected mass inside the three stance feet during lateral steps.
                support = 1. - np.where(swing, np.sin(np.pi * fraction), 0.)
                centroid = (self.foot_xy * support[:, None]).sum(axis=0) / support.sum()
                body_shift = (centroid - self.foot_xy.mean(axis=0)) * shift[0] * intensity
                delta[:, :2] -= body_shift
                delta[:, 2] -= shift[1] * intensity
            joints = np.einsum("lij,lj->il", self.inverse, delta).reshape(12)
            wheel = (vx + sides * (yaw_gain * yaw + yaw_feedback * (yaw - gyro_yaw))) / (.086 * 16.)
            return np.r_[np.clip(joints, -.65, .65) / .35, wheel].clip(-6., 6.)

        return act


def measure(params, cases, seeds, teacher, contract, limits, results):
    evaluation.sim.ROOT = results
    return [dict(evaluation.flat_episode(teacher.policy(params), contract, limits,
                                         np.asarray(evaluation.FLAT_COMMANDS[name]), seed),
                 command=name, seed=seed)
            for name in cases for seed in seeds]


def objective(params, teacher, contract, limits, results, cases):
    rows = measure(params, cases, [86201], teacher, contract, limits, results)
    return float(np.mean([r["xy_mae_m_s"] / .15 + r["yaw_mae_rad_s"] / .22
                          + r["stop_speed_m_s"] / .12 + r["stop_yaw_rate_rad_s"] / .15
                          + 5. * r["fallen"] for r in rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--lateral-only", action="store_true")
    parser.add_argument("--weight-shift", action="store_true",
                        help="Search a three-support-wheel lateral gait with body weight shift")
    args = parser.parse_args()
    evaluation.sim.ROOT = args.results.resolve()
    contract = json.loads((evaluation.sim.ROOT / "fall-baseline/fall-recovery-contract.json").read_text())
    limits = evaluation.sim.load_target_limits(contract)
    teacher = StepTeacher(contract)
    cases = LATERAL_CASES if args.lateral_only else CASES
    bounds = ((.62, .62), (.20, .44), (.05, .16), (.7, 3.2), *BOUNDS[4:]) if args.lateral_only else BOUNDS
    if args.weight_shift:
        if not args.lateral_only:
            parser.error("--weight-shift requires --lateral-only")
        bounds = ((.62, .62), (.16, .245), (.06, .17), (1.1, 4.),
                  *BOUNDS[4:], (.25, 1.8), (.0, .04))
    score = partial(objective, teacher=teacher, contract=contract, limits=limits,
                    results=evaluation.sim.ROOT, cases=cases)

    def save(intermediate_result):
        result = intermediate_result
        args.output.with_suffix(".progress.json").write_text(
            json.dumps({"params": result.x.tolist(), "training_cost": result.fun}) + "\n")
        print(f"generation={result.nit} cost={result.fun:.4f}", flush=True)

    result = differential_evolution(score, bounds, workers=args.workers, updating="deferred",
                                    maxiter=args.generations, popsize=4, polish=False,
                                    seed=420, callback=save)
    dev = measure(result.x, cases, range(86301, 86305), teacher, contract, limits, evaluation.sim.ROOT)
    args.output.write_text(json.dumps({"params": result.x.tolist(), "training_cost": result.fun,
                                      "dev": dev}, indent=2) + "\n")
    print(json.dumps({"validation_passed": sum(r["passed"] for r in dev), "trials": len(dev)}))


if __name__ == "__main__":
    main()
