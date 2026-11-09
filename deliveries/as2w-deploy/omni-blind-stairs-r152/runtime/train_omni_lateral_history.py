"""Distill lateral motion into a command-conditioned proprioceptive history head."""

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import evaluate_omni_stairs as evaluation
from screen_omni_contact_routing import ContactRoutedActor
from search_lateral_omni_teacher import LateralFeedbackTeacher, course
from train_recovery_student import make_actor, numpy_actor


CASES = ("left", "right", "diagonal_left", "diagonal_right")
MIRROR = np.arange(16) ^ 1
SCALE = np.r_[1.6, 2.8, 1.25, np.full(3, .25), np.ones(4),
              np.full(16, .8), np.full(16, .12), np.full(16, .5)].astype(np.float32)
FRAMES = (0, 2, 4, 6)
WIDTH = len(FRAMES) * len(SCALE) + 3


def history_head():
    return nn.Sequential(nn.Linear(WIDTH, 128), nn.ELU(), nn.Linear(128, 96), nn.ELU(), nn.Linear(96, 16))


def reflect_action(action):
    out = action[MIRROR].copy()
    out[:4] *= -1.
    return out


class History:
    def __init__(self):
        self.frames = deque(maxlen=7)
        self.ticks = 0
        self.active_side = 0
        self.previous_period = None
        self.phase_offset = 0.
        self.phase_tick = 0.

    def observe(self, obs, target, dt, period):
        frame = np.asarray(obs[:58], dtype=np.float32)
        if not self.frames:
            self.frames.extend([frame] * 7)
        else:
            self.frames.append(frame)
        side = int(np.sign(target[1])) if abs(target[1]) > .01 else 0
        if not side:
            self.active_side = 0
            self.ticks = 0
            self.previous_period = None
            self.phase_offset = 0.
            return None, False
        if side != self.active_side:
            self.ticks = 0
            self.active_side = side
            self.previous_period = None
            self.phase_offset = 0.
        frames = np.stack([self.frames[index] for index in FRAMES])
        mirrored = side < 0
        if mirrored:
            frames[:, [1, 2, 3, 5, 7]] *= -1.
            for start in (10, 26, 42):
                frames[:, start:start + 16] = frames[:, start + MIRROR]
                frames[:, start:start + 4] *= -1.
        elapsed = self.ticks * dt
        if self.previous_period is not None and self.previous_period != period:
            self.phase_offset += elapsed * (1. / self.previous_period - 1. / period)
        self.previous_period = period
        self.phase_tick = self.ticks + self.phase_offset * period / dt
        phase = 2. * np.pi * elapsed / period + 2. * np.pi * self.phase_offset
        self.ticks += 1
        return np.r_[(frames * SCALE).reshape(-1), np.sin(phase), np.cos(phase), min(elapsed / 4., 1.)], mirrored


def gait(target, params):
    weight = np.clip((abs(target[0]) - .06) / .25, 0., 1.)
    start = 2 if len(params) == 4 and target[1] < 0. else 0
    return (1. - weight) * params[start] + weight * params[start + 1]


class LateralController:
    def __init__(self, base, student, params, dt, teacher=None, teacher_fraction=0.):
        self.base = evaluation.prior_flat_planner(base, dt)
        self.student, self.params, self.dt = student, params, dt
        self.teacher, self.teacher_fraction = teacher, teacher_fraction
        self.history = History()
        self.samples = []

    def plan(self, model, data, obs, target, tick, advance):
        params = gait(target, self.params)
        features, mirrored = self.history.observe(obs, target, self.dt, params[0])
        if features is None:
            return self.base(model, data, obs, target, tick, advance)
        strength = min(1., abs(target[1]) / .14)
        if strength < 1.:
            baseline = self.base(model, data, obs, target, tick, advance)
        else:
            baseline = None
        if self.teacher is not None:
            teacher_action = self.teacher.act(model, data, obs, target, tick, advance)
            label = reflect_action(teacher_action) if mirrored else teacher_action
            self.samples.append((features, label.copy()))
        if self.student is None:
            action = teacher_action
        else:
            action = self.student(features)
            if mirrored:
                action = reflect_action(action)
            if self.teacher is not None:
                action = (1. - self.teacher_fraction) * action + self.teacher_fraction * teacher_action
        return action if baseline is None else (1. - strength) * baseline + strength * action


def archive_samples(paths, params, dt):
    samples = []
    for path in paths:
        with np.load(path) as data:
            for seed in np.unique(data["seed"]):
                mask = data["seed"] == seed
                history = History()
                for obs, action in zip(data["observations"][mask], data["actions"][mask]):
                    target = obs[:3] + (.4, 0., 0.)
                    features, mirrored = history.observe(obs, target, dt, gait(target, params)[0])
                    if features is not None:
                        samples.append((features, reflect_action(action) if mirrored else action.copy()))
    return samples


def development_cases():
    rng = np.random.default_rng(102381)
    cases = [(np.asarray(evaluation.FLAT_COMMANDS[name]), 102401 + 11 * index + replica)
             for index, name in enumerate(CASES) for replica in range(3)]
    for index in range(10):
        target = rng.uniform((-.6, .13, -.8), (.6, .35, .8))
        target[1] *= (-1.) ** index
        cases.append((target, 102501 + index))
    return cases


def load_actor(path):
    state = torch.load(path, weights_only=True, map_location="cpu")["model_state_dict"]
    return state, numpy_actor(make_actor(state))


def screen(base_factory, state, head, params, contract, limits, controller_factory=None, confirmation=False):
    sim = evaluation.sim
    dt = contract["control_dt"]
    defaults = np.asarray(contract["default_joint_pos"])
    if controller_factory is None:
        controller_factory = lambda base: LateralController(base, head, params, dt)
    flat = []
    if confirmation:
        rng = np.random.default_rng(91291)
        targets = rng.uniform((-.6, -.35, -.8), (.6, .35, .8), size=(48, 3))
        cases = [(f"mixed_{i:02d}", target, 91101 + i) for i, target in enumerate(targets)]
        cases.extend(("stop", np.zeros(3), seed) for seed in range(91201, 91205))
        stair_seeds = range(28001, 28065)
    else:
        cases = [(name, target, seed) for name, target in evaluation.FLAT_COMMANDS.items()
                 for seed in range(81201, 81205)]
        stair_seeds = range(27001, 27017)
    for name, target, seed in cases:
        base = base_factory()
        controller = controller_factory(base)
        row = evaluation.flat_episode(None, contract, limits, np.asarray(target), seed,
                                      evaluation.gait_clock, controller.plan)
        flat.append(dict(row, command=name, target=list(target), seed=seed,
                         settled_tick=getattr(base, "settled_tick", None)))
    stairs = []
    for seed in stair_seeds:
        model = sim.load_flat_model(True, True, "stairs", seed)
        stress = sim.sample_dynamics_stress(seed, model.nbody - 1, 16)
        base = base_factory()
        controller = controller_factory(base)
        command = np.array((.4, 0., 0.))

        def planner(data, obs, elapsed, advance):
            action = controller.plan(model, data, obs, command, round(elapsed / dt), advance)
            legs = np.clip(defaults[:12] + .35 * action[:12], *limits[:12, :2].T)
            wheels = np.clip(16. * action[12:], -limits[12:, 2], limits[12:, 2])
            return action, legs, wheels

        with torch.inference_mode():
            row = sim.evaluate(model, contract, state, seed, 0., limits=limits, action_clip=6.,
                               wheel_scale=16., wheel_angle_period=4. * np.pi, dynamics_stress=stress,
                               terrain="stairs", observation_suffix=evaluation.gait_clock, native_planner=planner)
        stairs.append(dict(row, passed=bool(row["passed"]), seed=seed,
                           switch_tick=getattr(base, "switch_tick", None),
                           settled_tick=getattr(base, "settled_tick", None)))
    return {"flat": flat, "stairs": stairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("anchor", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--teacher-parameters", type=Path, nargs=2, required=True)
    parser.add_argument("--teacher-demos", type=Path, nargs="+", required=True)
    parser.add_argument("--stair-head", type=Path)
    parser.add_argument("--contact-gate", type=Path)
    args = parser.parse_args()
    if bool(args.stair_head) != bool(args.contact_gate):
        parser.error("Provide both the stair head and the contact gate")
    torch.set_num_threads(1)
    torch.manual_seed(487)
    rng = np.random.default_rng(487)
    contract, limits, geometry = course(args.results.resolve())
    dt = contract["control_dt"]
    params = np.asarray([json.loads(path.read_text())["params"] for path in args.teacher_parameters])
    state, anchor = load_actor(args.anchor)
    if args.contact_gate:
        gate = torch.load(args.contact_gate, weights_only=True, map_location="cpu")
        stair_state, stair = load_actor(args.stair_head)
        base_factory = lambda: ContactRoutedActor(gate, anchor, stair, dt)
    else:
        gate = stair_state = None
        base_factory = lambda: anchor
    args.out.mkdir(parents=True, exist_ok=True)
    head = history_head()
    optimizer = torch.optim.Adam(head.parameters(), lr=6e-4)
    seeds = archive_samples(args.teacher_demos, params, dt)
    dataset = []
    best = None
    for round_index, teacher_fraction in enumerate((1., .78, .58, .40, .25, .12)):
        student = numpy_actor(head) if round_index else None
        for index in range(22):
            if index < 12:
                target = np.asarray(evaluation.FLAT_COMMANDS[CASES[index % len(CASES)]])
            else:
                target = rng.uniform((-.55, .11, -.7), (.55, .35, .7))
                target[1] *= (-1.) ** index
            teacher = LateralFeedbackTeacher(contract, geometry, gait(target, params))
            controller = LateralController(base_factory(), student, params, dt, teacher, teacher_fraction)
            row = evaluation.flat_episode(None, contract, limits, target, 101001 + 100 * round_index + index,
                                          evaluation.gait_clock, controller.plan)
            if not row["fallen"]:
                dataset.extend(controller.samples)
        (demo_x, demo_y), (rollout_x, rollout_y) = [
            tuple(torch.as_tensor(np.stack(values), dtype=torch.float32) for values in zip(*source))
            for source in (seeds, dataset or seeds)
        ]
        for _ in range(210):
            demo_index = torch.randint(len(demo_x), (96,))
            roll_index = torch.randint(len(rollout_x), (192,))
            x = torch.cat((demo_x[demo_index], rollout_x[roll_index]))
            y = torch.cat((demo_y[demo_index], rollout_y[roll_index]))
            loss = nn.functional.smooth_l1_loss(head(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        rows = []
        for target, seed in development_cases():
            controller = LateralController(base_factory(), numpy_actor(head), params, dt)
            rows.append(evaluation.flat_episode(None, contract, limits, target, seed,
                                                evaluation.gait_clock, controller.plan))
        score = (sum(row["passed"] for row in rows[:12]), sum(row["passed"] for row in rows[12:]),
                 -sum(row["xy_mae_m_s"] / .15 + row["yaw_mae_rad_s"] / .22 for row in rows))
        print(json.dumps({"round": round_index, "loss": loss.item(), "score": score,
                          "rollout_frames": len(dataset)}), flush=True)
        if best is None or score > best:
            best = score
            torch.save({"anchor": state, "stair": stair_state, "contact_gate": gate,
                        "lateral": head.state_dict(), "teacher_period_parameters": params.tolist()},
                       args.out / "policy.pt")
    selected = torch.load(args.out / "policy.pt", weights_only=True, map_location="cpu")
    head.load_state_dict(selected["lateral"])
    report = screen(base_factory, state, numpy_actor(head), params, contract, limits)
    (args.out / "evaluation.json").write_text(json.dumps(report, indent=2, default=lambda v: v.item()) + "\n")
    print(json.dumps({name: [sum(row["passed"] for row in rows), len(rows)] for name, rows in report.items()}))


if __name__ == "__main__":
    main()
