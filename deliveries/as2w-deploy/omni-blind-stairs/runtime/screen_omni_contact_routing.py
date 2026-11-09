"""Screen a single proprioception-routed actor on flat commands and blind stairs."""

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_omni_stairs as evaluation
from train_omni_stair_contact_gate import FEATURES, FRAME_INDICES, HISTORY, contact_gate
from train_recovery_student import make_actor, numpy_actor


class ContactRoutedActor:
    def __init__(self, gate, anchor, stair, dt):
        self.gate = contact_gate().eval()
        self.gate.load_state_dict(gate["gate_state_dict"])
        self.center, self.scale, self.threshold = (gate[key] for key in ("center", "scale", "threshold"))
        self.required_consecutive = gate.get("required_consecutive", 3)
        self.blend_seconds = gate.get("blend_seconds", .18)
        self.anchor, self.stair, self.dt = anchor, stair, dt
        self.history = deque(maxlen=HISTORY)
        self.tick = 0
        self.consecutive = 0
        self.switch_tick = None

    @torch.inference_mode()
    def __call__(self, obs):
        frame = obs[FEATURES].astype(np.float32, copy=False)
        if not self.history:
            self.history.extend([frame] * HISTORY)
        else:
            self.history.append(frame)
        if self.switch_tick is None:
            features = torch.from_numpy(np.concatenate([self.history[index] for index in FRAME_INDICES]))
            normalized = ((features - self.center) / self.scale).clamp(-8., 8.)
            active = float(self.gate(normalized).sigmoid()) >= self.threshold
            self.consecutive = self.consecutive + 1 if active else 0
            if self.consecutive >= self.required_consecutive:
                self.switch_tick = self.tick
        anchor = self.anchor(obs)
        if self.switch_tick is not None:
            blend = min(1., (self.tick - self.switch_tick + 1) * self.dt / self.blend_seconds)
            anchor = (1. - blend) * anchor + blend * self.stair(obs)
        self.tick += 1
        return anchor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("anchor", type=Path)
    parser.add_argument("stair", type=Path)
    parser.add_argument("gate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    sim = evaluation.sim
    sim.ROOT = args.results.resolve()
    contract = json.loads((sim.ROOT / "fall-baseline/fall-recovery-contract.json").read_text())
    limits = sim.load_target_limits(contract)
    defaults = np.asarray(contract["default_joint_pos"])
    anchor_state = torch.load(args.anchor, map_location="cpu", weights_only=True)["model_state_dict"]
    stair_state = torch.load(args.stair, map_location="cpu", weights_only=True)["model_state_dict"]
    gate = torch.load(args.gate, map_location="cpu", weights_only=True)
    anchor, stair = (numpy_actor(make_actor(state)) for state in (anchor_state, stair_state))
    dt = contract["control_dt"]
    flat, stairs = [], []
    for command, target in evaluation.FLAT_COMMANDS.items():
        for seed in range(81201, 81205):
            actor = ContactRoutedActor(gate, anchor, stair, dt)
            planner = evaluation.prior_flat_planner(actor, dt)
            row = evaluation.flat_episode(actor, contract, limits, np.asarray(target), seed,
                                          evaluation.gait_clock, planner)
            flat.append(dict(row, command=command, seed=seed, switch_tick=actor.switch_tick))

    with torch.inference_mode():
        for seed in range(27001, 27017):
            actor = ContactRoutedActor(gate, anchor, stair, dt)
            model = sim.load_flat_model(True, True, "stairs", seed)
            stress = sim.sample_dynamics_stress(seed, model.nbody - 1, 16)

            def plan(model_data, obs, elapsed, advance):
                action = actor(obs)
                leg = np.clip(defaults[:12] + .35 * action[:12], *limits[:12, :2].T)
                wheel = np.clip(16. * action[12:], -limits[12:, 2], limits[12:, 2])
                return action, leg, wheel

            row = sim.evaluate(
                model, contract, anchor_state, seed, 0., limits=limits, action_clip=6.,
                wheel_scale=16., wheel_angle_period=4. * np.pi, dynamics_stress=stress,
                terrain="stairs", observation_suffix=evaluation.gait_clock, native_planner=plan,
            )
            stairs.append(dict(row, passed=bool(row["passed"]), seed=seed, switch_tick=actor.switch_tick))
    report = {"flat": flat, "stairs": stairs}
    args.output.write_text(json.dumps(report, indent=2, default=lambda value: value.item()) + "\n")
    print(json.dumps({
        scene: {"passed": sum(row["passed"] for row in rows),
                "switched": sum(row["switch_tick"] is not None for row in rows), "trials": len(rows)}
        for scene, rows in report.items()
    }), flush=True)


if __name__ == "__main__":
    main()
