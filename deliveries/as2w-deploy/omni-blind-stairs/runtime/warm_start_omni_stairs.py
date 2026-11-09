"""Combine verified flat driving demonstrations with stair teacher replays."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_omni_stairs as evaluation
from train_recovery_student import make_actor, numpy_actor


def stair_data(paths, contract, limits, remap_gains, clock=False):
    model = evaluation.sim.load_flat_model(True, True)
    joints = [model.joint(name) for name in contract["joint_names"]]
    q_ids = [joint.qposadr[0] for joint in joints]
    dq_ids = [model.nq + joint.dofadr[0] for joint in joints]
    defaults = np.asarray(contract["default_joint_pos"])
    inputs, targets = [], []
    for path in paths:
        with np.load(path) as replay:
            state = replay["state"]
            control = replay["control"]
        if state.shape != (len(control), model.nq + model.nv) or control.shape[1:] != (16,):
            raise ValueError(f"Unexpected replay layout in {path}")
        q = state[:, q_ids].copy()
        q[:, 12:] = (q[:, 12:] + 2. * np.pi) % (4. * np.pi) - 2. * np.pi
        rotations = np.empty((len(state), 9))
        for index, quat in enumerate(state[:, 3:7]):
            evaluation.sim.mj.mju_quat2Mat(rotations[index], quat)
        leg_target = control[:, :12]
        if remap_gains:
            # Match the archived teacher's leg torque at the sampled state.
            # Its wheel slip controller is not recoverable from target-only traces.
            leg_q, leg_dq = q[:, :12], state[:, dq_ids[:12]]
            leg_target = leg_q + ((104. * (leg_target - leg_q))
                                  - (3. * np.sqrt(1.3) - 2.) * leg_dq) / 55.
            leg_target = np.clip(leg_target, limits[:12, 0], limits[:12, 1])
        action = np.column_stack(((leg_target - defaults[:12]) / .35, control[:, 12:] / 16.))
        action = action.clip(-6, 6)
        previous = np.vstack((np.zeros(16), action[:-1]))
        obs = np.column_stack((np.zeros((len(state), 3)), state[:, model.nq + 3:model.nq + 6],
                               -rotations[:, 6:], np.full(len(state), .4), q - defaults,
                               state[:, dq_ids], previous))
        if clock:
            obs = np.column_stack((obs, *evaluation.gait_clock(np.arange(len(state)) * contract["control_dt"])))
        inputs.append(obs)
        targets.append(action)
    return np.concatenate(inputs), np.concatenate(targets)


def flat_data(teacher, contract, limits, first_seed, trials, behavior=None, clock=None):
    inputs, targets = [], []
    speeds = (0., .2, -.2, .4, -.4, .6, -.6)
    accepted = set()
    for trial in range(trials):
        speed = speeds[trial % len(speeds)]
        samples = []

        def rolling_actor(obs):
            action = teacher(obs[:58])
            action[12:] = np.clip((obs[0] + .4) / (.086 * 16.), -6, 6)
            samples.append((obs.copy(), action.copy()))
            return action if behavior is None else behavior(obs)

        result = evaluation.flat_episode(rolling_actor, contract, limits,
                                         np.array([speed, 0., 0.]), first_seed + trial, clock)
        if result["passed"] or behavior is not None:
            x, y = map(np.stack, zip(*samples))
            inputs.append(x)
            targets.append(y)
            accepted.add(speed)
    if 0. not in accepted or not any(x > 0 for x in accepted) or not any(x < 0 for x in accepted):
        raise RuntimeError(f"Flat teacher lacked valid stop, forward or reverse examples: {accepted}")
    return (np.concatenate(inputs), np.concatenate(targets)), len(inputs), sorted(accepted)


def stepping_data(reports, contract, limits, first_seed, trials, behavior=None):
    from search_flat_omni_teacher import StepTeacher

    teacher = StepTeacher(contract)
    inputs, targets, accepted = [], [], []
    for path in reports:
        report = json.loads(path.read_text())
        params = report["params"]
        params[0] = .62
        commands = sorted({row["command"] for row in report["dev"] if row["passed"]})
        for name in commands:
            for seed in range(first_seed, first_seed + trials):
                policy = teacher.policy(params)
                samples = []

                def record(obs):
                    action = policy(obs)
                    samples.append((obs.copy(), action.copy()))
                    return action if behavior is None else behavior(obs)

                target = np.asarray(evaluation.FLAT_COMMANDS[name])
                result = evaluation.flat_episode(record, contract, limits, target, seed, evaluation.gait_clock)
                if result["passed"] or behavior is not None:
                    x, y = map(np.stack, zip(*samples))
                    inputs.append(x)
                    targets.append(y)
                    accepted.append(name)
    if not inputs:
        raise RuntimeError("No stepping teacher demonstrations passed after synchronizing the gait clock")
    return (np.concatenate(inputs), np.concatenate(targets)), accepted


def benchmark_data(path, control_dt, clock):
    with np.load(path) as archive:
        obs, actions, seeds = (archive[name] for name in ("observations", "actions", "seed"))
    if obs.shape != (len(seeds), 58) or actions.shape != (len(seeds), 16):
        raise ValueError("Unexpected benchmark teacher observation or action shape")
    unique = np.unique(seeds)
    if not len(unique) or not np.isfinite(obs).all() or not np.isfinite(actions).all():
        raise ValueError("Benchmark teacher archive is empty or nonfinite")
    if clock:
        elapsed = np.empty(len(seeds))
        for seed in unique:
            episode = seeds == seed
            elapsed[episode] = np.arange(episode.sum()) * control_dt
        obs = np.column_stack((obs, *clock(elapsed)))
    dev = seeds == unique[-1] if len(unique) > 1 else np.zeros(len(seeds), dtype=bool)
    return (obs[~dev], actions[~dev]), (obs[dev], actions[dev]), unique.tolist()


def tensor_data(data, mirror=False):
    x, y = data
    if mirror:
        indices = np.arange(16) ^ 1
        signs = np.r_[np.full(4, -1.), np.ones(12)]
        reflected = x.copy()
        reflected[:, [1, 2, 3, 5, 7]] *= -1
        if reflected.shape[1] == 60:
            reflected[:, -2:] *= -1
        for start in (10, 26, 42):
            reflected[:, start:start + 16] = x[:, start + indices] * signs
        x, y = np.concatenate((x, reflected)), np.concatenate((y, y[:, indices] * signs))
    return tuple(torch.as_tensor(v, dtype=torch.float32) for v in (x, y))


def motion_prior_anchor(actor, contract, limits):
    accepted = []
    commands = []
    with torch.inference_mode():
        for name, target in evaluation.FLAT_COMMANDS.items():
            for seed in (86801, 86802):
                samples = []
                planner = evaluation.prior_flat_planner(
                    actor, contract["control_dt"], lambda obs, action: samples.append((obs.copy(), action.copy())),
                )
                result = evaluation.flat_episode(
                    actor, contract, limits, np.asarray(target), seed, evaluation.gait_clock, planner,
                )
                if result["passed"]:
                    accepted.extend(samples)
                    commands.append(name)
    if not accepted:
        raise ValueError("Initial actor did not complete any flat anchor episodes")
    return tensor_data(tuple(map(np.stack, zip(*accepted)))), sorted(set(commands))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("initial", type=Path)
    parser.add_argument("flat_teacher", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stair-train", type=Path, nargs="+", required=True)
    parser.add_argument("--stair-validation", type=Path, nargs="+", required=True)
    parser.add_argument("--student-rollouts", action="store_true",
                        help="Correct the teacher targets on states visited by the student")
    parser.add_argument("--stair-leg-gain-remap", action="store_true",
                        help="Convert archived R17 leg targets to the deployment PD gains")
    parser.add_argument("--step-teachers", nargs="+", type=Path,
                        help="Add validated lateral and steering demonstrations to a 60-input actor")
    parser.add_argument("--benchmark-teacher", type=Path, nargs="+",
                        help="Add independent archives of completed five-step benchmark demonstrations")
    parser.add_argument("--motion-prior-anchor-weight", type=float, default=0.,
                        help="Preserve initial actor actions from successful flat rollouts with the gait prior")
    args = parser.parse_args()
    if set(p.resolve() for p in args.stair_train) & set(p.resolve() for p in args.stair_validation):
        parser.error("Stair train and validation trajectories must be disjoint")
    torch.set_num_threads(1)
    torch.manual_seed(419)
    evaluation.sim.ROOT = args.results.resolve()
    contract = json.loads((evaluation.sim.ROOT / "fall-baseline/fall-recovery-contract.json").read_text())
    limits = evaluation.sim.load_target_limits(contract)
    source = torch.load(args.initial, weights_only=True, map_location="cpu")["model_state_dict"]
    clock = evaluation.gait_clock if source["actor.0.weight"].shape[1] == 60 else None
    if args.motion_prior_anchor_weight < 0. or (args.motion_prior_anchor_weight and clock is None):
        parser.error("The motion prior anchor requires a nonnegative weight and a 60-input actor")
    if args.step_teachers and clock is None:
        parser.error("--step-teachers requires a 60-input checkpoint with a gait clock")
    teacher_state = torch.load(args.flat_teacher, weights_only=True, map_location="cpu")["model_state_dict"]
    flat_teacher = numpy_actor(make_actor(teacher_state))
    train_flat, accepted, speeds = flat_data(flat_teacher, contract, limits, 85001, 28, clock=clock)
    validation_flat, accepted_dev, speeds_dev = flat_data(flat_teacher, contract, limits, 85101, 14, clock=clock)
    stair_train = tensor_data(stair_data(args.stair_train, contract, limits, args.stair_leg_gain_remap, clock), True)
    train = [tensor_data(train_flat, True), stair_train]
    stair_dev = stair_data(args.stair_validation, contract, limits, args.stair_leg_gain_remap, clock)
    validation = [tensor_data(validation_flat), tensor_data(stair_dev)]
    steering = []
    if args.step_teachers:
        stepping, steering = stepping_data(args.step_teachers, contract, limits, 85401, 4)
        stepping_dev, _ = stepping_data(args.step_teachers, contract, limits, 85501, 2)
        train.append(tensor_data(stepping, True))
        validation.append(tensor_data(stepping_dev))
    benchmark_seeds = []
    for archive in args.benchmark_teacher or ():
        course_train, course_dev, seeds = benchmark_data(archive, contract["control_dt"], clock)
        if set(seeds) & set(benchmark_seeds):
            parser.error("Benchmark teacher archives must use disjoint episode seeds")
        benchmark_seeds.extend(seeds)
        train.append(tensor_data(course_train, True))
        if len(course_dev[0]):
            validation.append(tensor_data(course_dev))
    actor = make_actor(source)
    anchor = None
    anchor_commands = []
    if args.motion_prior_anchor_weight:
        anchor, anchor_commands = motion_prior_anchor(numpy_actor(actor), contract, limits)
        print(json.dumps({"motion_prior_anchor_commands": anchor_commands}), flush=True)
    optimizer = torch.optim.Adam(actor.parameters(), lr=2e-4)
    criterion = torch.nn.SmoothL1Loss()
    best = None
    iterations = 700 if args.student_rollouts else 1000
    for step in range(iterations):
        if args.student_rollouts and step in (0, 300, 500):
            rollout_data, _, _ = flat_data(flat_teacher, contract, limits, 85201 + step, 14,
                                            behavior=numpy_actor(actor), clock=clock)
            train_flat = tuple(np.concatenate((old, new)) for old, new in zip(train_flat, rollout_data))
            train[0] = tensor_data(train_flat, True)
            if args.step_teachers:
                steering_rollouts, _ = stepping_data(args.step_teachers, contract, limits,
                                                     85601 + step, 3, behavior=numpy_actor(actor))
                stepping = tuple(np.concatenate((old, new)) for old, new in zip(stepping, steering_rollouts))
                train[2] = tensor_data(stepping, True)
        losses = []
        for x, y in train:
            sample = torch.randint(len(x), (256,))
            losses.append(criterion(actor(x[sample]).clamp(-6, 6), y[sample]))
        if anchor is not None:
            x, y = anchor
            sample = torch.randint(len(x), (256,))
            losses.append(args.motion_prior_anchor_weight * criterion(actor(x[sample]), y[sample]))
        optimizer.zero_grad()
        sum(losses).backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            with torch.inference_mode():
                values = [float(criterion(actor(x).clamp(-6, 6), y)) for x, y in validation]
            closed_loop = None
            if args.student_rollouts:
                behavior = numpy_actor(actor)
                closed_loop = sum(evaluation.flat_episode(
                    behavior, contract, limits, np.array([speed, 0., 0.]), 85301 + index, clock)["passed"]
                    for index, speed in enumerate(speeds_dev))
                closed_loop += sum(evaluation.flat_episode(
                    behavior, contract, limits, np.asarray(evaluation.FLAT_COMMANDS[name]),
                    85321 + index, clock)["passed"] for index, name in enumerate(sorted(set(steering))))
            print(json.dumps({"update": step + 1, "flat_loss": values[0], "stair_loss": values[1],
                              "flat_demos": accepted, "flat_dev_demos": accepted_dev,
                              "closed_loop_dev_passed": closed_loop}), flush=True)
            score = (closed_loop or 0, -sum(values))
            if best is None or score > best:
                best = score
                state = dict(source)
                state.update({f"actor.{key}": value for key, value in actor.state_dict().items()})
                torch.save({"model_state_dict": state, "iter": 0, "infos": {
                    "flat_demo_speeds": speeds, "flat_validation_speeds": speeds_dev,
                    "flat_loss": values[0], "stair_loss": values[1],
                    "stair_train": [path.name for path in args.stair_train],
                    "stair_validation": [path.name for path in args.stair_validation],
                    "student_rollouts": args.student_rollouts,
                    "stair_leg_gain_remap": args.stair_leg_gain_remap,
                    "closed_loop_dev_passed": closed_loop,
                    "stepping_demo_commands": sorted(set(steering)),
                    "benchmark_teacher_seeds": benchmark_seeds,
                    "motion_prior_anchor_commands": anchor_commands,
                }}, args.output)


if __name__ == "__main__":
    main()
