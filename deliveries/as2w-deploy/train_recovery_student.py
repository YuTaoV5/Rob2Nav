"""Distill a privileged recovery actor into a proprioception-only student."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import mujoco_screen as sim


def make_actor(state):
    layers = []
    for index in (0, 2, 4, 6):
        weight = state[f"actor.{index}.weight"]
        outputs, inputs = weight.shape
        layers.append(nn.Linear(inputs, outputs))
        if index != 6:
            layers.append(nn.ELU())
    actor = nn.Sequential(*layers)
    actor.load_state_dict({key.removeprefix("actor."): value for key, value in state.items()
                           if key.startswith("actor.")})
    return actor


def numpy_actor(actor):
    layers = [(layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy())
              for layer in actor if isinstance(layer, nn.Linear)]

    def infer(obs):
        values = obs.astype(np.float32, copy=False)
        for index, (weight, bias) in enumerate(layers):
            values = weight @ values + bias
            if index < len(layers) - 1:
                negative = values < 0
                values[negative] = np.expm1(values[negative])
        return np.clip(values, -6, 6)

    return infer


def rollout(contract, limits, state, pose, seed, terrain, disturbed, teacher, actor, records=None):
    model = sim.load_flat_model(True, True, terrain, seed)
    stress = sim.sample_dynamics_stress(seed, model.nbody - 1, 16) if disturbed else None
    defaults = np.asarray(contract["default_joint_pos"])
    period = 4 * np.pi

    def policy(raw, elapsed):
        raw[22:26] = (raw[22:26] + period / 2) % period - period / 2
        raw[10:26] -= defaults
        active = elapsed >= contract["settle_seconds"]
        label = teacher(raw) if teacher is not None else None
        if actor is not None:
            raw[:3] = 0
        if records is not None and elapsed >= contract["settle_seconds"] - .1:
            records.append((raw, label))
        action = label if actor is None else actor(raw)
        return (action, np.clip(defaults[:12] + .35 * action[:12], *limits[:12, :2].T),
                np.clip(16 * action[12:], -limits[12:, 2], limits[12:, 2]), active)

    return sim.evaluate(model, contract, state, seed, sim.POSES[pose], limits=limits,
                        action_clip=6., wheel_scale=16., wheel_angle_period=period,
                        dynamics_stress=stress, terrain=terrain, native_policy=policy)


def fit(checkpoint, teacher, student, contract, limits, output, device, rounds, focus):
    state = checkpoint["model_state_dict"]
    optimizer = torch.optim.Adam(student.parameters(), lr=8e-5 if focus else 3e-4)
    generator = np.random.default_rng(413)
    terrains = {
        None: (None, None, "slope", "rough"),
        "slope": (None, "slope", "slope", "rough"),
        "flat": (None, None, None, "slope", None, None, None, "rough"),
    }[focus]
    seed_start = 51000 if focus else 50000
    records = []
    best = (-1, -1.)
    for iteration in range(rounds):
        current = None if iteration == 0 and not focus else numpy_actor(student)
        for case in range(48):
            terrain = terrains[case // 4 % len(terrains)]
            pose = tuple(sim.POSES)[case % 4]
            first = len(records)
            rollout(contract, limits, state, pose, seed_start + 48 * iteration + case,
                    terrain, case % 3 != 0, teacher, current, records)
            if current is None:
                for obs, _ in records[first:]:
                    obs[:3] = 0

        inputs, targets = map(np.stack, zip(*records))
        inputs = torch.as_tensor(inputs, dtype=torch.float32, device=device)
        targets = torch.as_tensor(targets, dtype=torch.float32, device=device)
        for _ in range(350):
            batch = torch.as_tensor(generator.integers(len(inputs), size=512), device=device)
            prediction = student(inputs[batch]).clamp(-6, 6)
            loss = nn.functional.smooth_l1_loss(prediction, targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        current = numpy_actor(student)
        dev_cases = range(4 if focus else 2)
        dev = [
            rollout(contract, limits, state, pose, 61000 + trial, terrain, trial % 2 != 0, None, current)
            for terrain in (None, "slope", "rough") for pose in sim.POSES for trial in dev_cases
        ]
        score = sum(row["passed"] for row in dev), sum(row["stable_s"] for row in dev)
        print(json.dumps({"round": iteration + 1, "samples": len(records), "loss": loss.item(),
                          "dev_passed": score[0], "dev_trials": len(dev)}), flush=True)
        if score > best:
            best = score
            model = copy.copy(state)
            model.update({f"actor.{key}": value.detach().cpu() for key, value in student.state_dict().items()})
            model["actor.0.weight"][:, :3] = 0
            torch.save({"model_state_dict": model, "iter": 0, "infos": {
                "base_linear_velocity": "masked", "distillation_round": iteration + 1,
                "development_passed": score[0], "development_trials": len(dev), "terrain_focus": focus,
            }}, output)


SCREENS = {
    "standard": (range(12001, 12005), None, False),
    "confirmation": (range(22001, 22017), None, False),
    "flat-dynamics-control": (range(23001, 23017), None, False),
    "flat-dynamics-stress": (range(23001, 23017), None, True),
    "slope": (range(12001, 12005), "slope", True),
    "rough": (range(12001, 12005), "rough", True),
}


def evaluate(checkpoint, student_path, contract, limits, output, mode):
    state = checkpoint["model_state_dict"]
    if mode == "masked-teacher":
        state = dict(state)
        state["actor.0.weight"] = state["actor.0.weight"].clone()
        state["actor.0.weight"][:, :3] = 0
    elif mode == "student":
        state = torch.load(student_path, weights_only=True, map_location="cpu")["model_state_dict"]
        if torch.count_nonzero(state["actor.0.weight"][:, :3]):
            raise ValueError("Student actor depends on the velocity channels")

    with output.open("w", encoding="utf-8") as stream, torch.inference_mode():
        for name, (seeds, terrain, disturbed) in SCREENS.items():
            rows = []
            for pose in sim.POSES:
                for seed in seeds:
                    model = sim.load_flat_model(True, True, terrain, seed)
                    stress = sim.sample_dynamics_stress(seed, model.nbody - 1, 16) if disturbed else None
                    row = sim.evaluate(model, contract, state, seed, sim.POSES[pose], limits=limits,
                                       action_clip=6., wheel_scale=16., wheel_angle_period=4 * np.pi,
                                       dynamics_stress=stress, terrain=terrain)
                    row.update(screen=name, pose=pose, seed=seed)
                    rows.append(row)
                    stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(json.dumps({mode: {name: [sum(row["passed"] for row in rows), len(rows)]}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("student", type=Path)
    parser.add_argument("--evaluate", choices=("teacher", "masked-teacher", "student"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--initialize-student", type=Path)
    parser.add_argument("--focus", choices=("flat", "slope"))
    args = parser.parse_args()
    if args.rounds <= 0 or bool(args.focus) != bool(args.initialize_student):
        parser.error("Use positive rounds and supply --focus together with --initialize-student")
    torch.set_num_threads(1)
    sim.ROOT = args.results.resolve()
    contract = json.loads((sim.ROOT / "fall-baseline/fall-recovery-contract.json").read_text())
    limits = sim.load_target_limits(contract)
    checkpoint = torch.load(args.teacher, weights_only=True, map_location="cpu")
    student = make_actor(checkpoint["model_state_dict"])
    teacher = numpy_actor(student)
    if args.evaluate:
        if args.report is None:
            parser.error("--evaluate requires --report")
        evaluate(checkpoint, args.student, contract, limits, args.report, args.evaluate)
    else:
        if args.initialize_student:
            source = torch.load(args.initialize_student, weights_only=True, map_location="cpu")
            student = make_actor(source["model_state_dict"])
        else:
            student = copy.deepcopy(student)
        fit(checkpoint, teacher, student.to(args.device), contract, limits,
            args.student, args.device, args.rounds, args.focus)


if __name__ == "__main__":
    main()
