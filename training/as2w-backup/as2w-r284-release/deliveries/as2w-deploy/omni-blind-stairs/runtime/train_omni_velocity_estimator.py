"""Fit an onboard kinematic velocity correction and test it in closed loop."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import evaluate_omni_stairs as evaluation
from probe_omni_wheel_odometry import WheelOdometry, wheel_profiles
from search_lateral_omni_teacher import LateralFeedbackTeacher, course
from train_omni_lateral_history import CASES, WIDTH, History, gait, load_actor, reflect_action
from train_recovery_student import numpy_actor


def estimator_model():
    return nn.Sequential(nn.Linear(WIDTH + 2, 128), nn.ELU(), nn.Linear(128, 64),
                         nn.ELU(), nn.Linear(64, 2))


def run_case(target, seed, base, contract, limits, geometry, params, correction, privileged_fraction, profiles=()):
    dt = contract["control_dt"]
    history = History()
    odometry = WheelOdometry(contract["default_joint_pos"], profiles=profiles)
    teacher = LateralFeedbackTeacher(contract, geometry, gait(target, params))
    fallback = evaluation.prior_flat_planner(base, dt)
    samples = []
    errors = []
    base_id = None

    def plan(model, data, obs, command, tick, advance):
        nonlocal base_id
        if base_id is None:
            base_id = model.body("base_link").id
        velocity = odometry.update(obs)
        features, mirrored = history.observe(obs, command, dt, teacher.params[0])
        if features is None:
            return fallback(model, data, obs, command, tick, advance)
        reflection = np.array((1., -1., -1.)) if mirrored else np.ones(3)
        encoder = velocity * reflection[:2]
        features = np.r_[features, encoder]
        refined = encoder if correction is None else encoder + correction(features).clip(-.55, .55)
        rotation = data.xmat[base_id].reshape(3, 3)
        truth = (rotation.T @ data.qvel[:3])[:2] * reflection[:2]
        samples.append((features, (truth - encoder).clip(-.55, .55)))
        if tick * dt >= 1.5:
            errors.append((np.abs(refined - truth), np.abs(encoder - truth)))
        feedback = refined
        if privileged_fraction:
            feedback = (1. - privileged_fraction) * refined + privileged_fraction * truth
        command = command * reflection
        error = command - np.r_[feedback, obs[5] * reflection[2]]
        teacher.integral = (teacher.integral + dt * error).clip(-.50, .50)
        action = teacher.move(-obs[7] * reflection[1], command, tick, error)
        return reflect_action(action) if mirrored else action

    row = evaluation.flat_episode(None, contract, limits, target, seed, evaluation.gait_clock, plan)
    return dict(row, seed=seed, target=target.tolist(),
                velocity_mae=np.mean(errors, axis=0).tolist()), samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("anchor", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--teacher-parameters", type=Path, nargs=2, required=True)
    parser.add_argument("--mesh-odometry", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(504)
    rng = np.random.default_rng(504)
    contract, limits, geometry = course(args.results.resolve())
    profiles = wheel_profiles(evaluation.sim.load_flat_model(True, True), contract["joint_names"]) if args.mesh_odometry else ()
    anchor_state, base = load_actor(args.anchor)
    params = np.asarray([json.loads(path.read_text())["params"] for path in args.teacher_parameters])
    targets = [np.asarray(evaluation.FLAT_COMMANDS[name]) for name in CASES]
    model = estimator_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    samples = []
    best = None
    args.out.mkdir(parents=True, exist_ok=True)
    for index, privileged_fraction in enumerate((1., .7, .4, .15, 0.)):
        previous = numpy_actor(model) if index else None
        for i in range(28):
            if i < 16:
                target = targets[i % len(targets)]
            else:
                target = rng.uniform((-.60, .12, -.75), (.60, .35, .75))
                target[1] *= (-1.) ** i
            _, data = run_case(target, 109001 + 100 * index + i, base, contract, limits, geometry,
                               params, previous, privileged_fraction, profiles)
            samples.extend(data)
        x, y = (torch.as_tensor(np.stack(values), dtype=torch.float32) for values in zip(*samples))
        for _ in range(240):
            batch = torch.randint(len(x), (256,))
            loss = nn.functional.smooth_l1_loss(model(x[batch]), y[batch], beta=.08)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        actor = numpy_actor(model)
        dev = [run_case(target, 110301 + 12 * i + j, base, contract, limits, geometry, params, actor, 0., profiles)[0]
               for i, target in enumerate(targets) for j in range(3)]
        score = (sum(row["passed"] for row in dev),
                 -sum(row["xy_mae_m_s"] / .15 + row["yaw_mae_rad_s"] / .22 for row in dev))
        print(json.dumps({"round": index, "score": score, "samples": len(samples),
                          "velocity_mae": np.mean([row["velocity_mae"] for row in dev], axis=0).tolist()}), flush=True)
        if best is None or score > best:
            best = score
            torch.save({"estimator": model.state_dict(), "anchor": anchor_state, "parameters": params.tolist(),
                        "odometry_profiles": [(torch.as_tensor(a), torch.as_tensor(r)) for a, r in profiles]},
                       args.out / "policy.pt")
    selected = torch.load(args.out / "policy.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(selected["estimator"])
    actor = numpy_actor(model)
    rows = [run_case(target, 111201 + 16 * i + j, base, contract, limits, geometry, params, actor, 0., profiles)[0]
            for i, target in enumerate(targets) for j in range(8)]
    (args.out / "evaluation.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(json.dumps({"passed": sum(row["passed"] for row in rows), "trials": len(rows),
                      "velocity_mae": np.mean([row["velocity_mae"] for row in rows], axis=0).tolist()}), flush=True)


if __name__ == "__main__":
    main()
