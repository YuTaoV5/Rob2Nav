"""Collect successful GRU-switched stair demonstrations for SuperMimic replay."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_omni_stairs as evaluation
import mujoco_screen as sim
from train_recovery_student import make_actor, numpy_actor
from train_top_detector import TopDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("detector", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, default=.90)
    parser.add_argument("--streak", type=int, default=3)
    parser.add_argument("--max-trigger-time", type=float, default=8.)
    parser.add_argument("--wheel-action", type=float, default=.10)
    parser.add_argument("--stair-rise", type=float, default=.06)
    args = parser.parse_args()

    torch.set_num_threads(1)
    sim.ROOT = args.results.resolve()
    contract = json.loads((sim.ROOT / "fall-baseline/fall-recovery-contract.json").read_text())
    limits = sim.load_target_limits(contract)
    state = torch.load(args.checkpoint, weights_only=True, map_location="cpu")["model_state_dict"]
    actor = numpy_actor(make_actor(state))
    default = np.asarray(contract["default_joint_pos"])
    detector_state = torch.load(args.detector, weights_only=True, map_location="cpu")
    detector = TopDetector(
        np.zeros(detector_state["input_size"], dtype=np.float32),
        np.ones(detector_state["input_size"], dtype=np.float32),
        detector_state["hidden_size"],
    )
    detector.load_state_dict(detector_state["model_state_dict"])
    detector.eval()
    observations = []
    actions = []
    episode_seeds = []
    report = []
    seeds = [*range(27001, 27017), *range(31001, 31017)]
    with torch.inference_mode():
        for seed in seeds:
            model = sim.load_flat_model(True, True, "stairs", seed, stair_rise=args.stair_rise)
            stress = sim.sample_dynamics_stress(seed, model.nbody - 1, 16)
            hidden = None
            consecutive = 0
            switched = False
            samples = []

            def planner(data, obs, elapsed, advance):
                nonlocal hidden, consecutive, switched
                logits, hidden = detector(
                    torch.as_tensor(obs, dtype=torch.float32)[None, None], hidden,
                )
                probability = float(torch.sigmoid(logits)[0, 0])
                consecutive = consecutive + 1 if probability >= args.threshold else 0
                if (not switched and elapsed <= args.max_trigger_time
                        and consecutive >= args.streak):
                    switched = True
                action = (np.r_[np.zeros(12), np.full(4, args.wheel_action)]
                          if switched else np.clip(actor(obs), -6., 6.))
                samples.append((obs[:58].astype(np.float32, copy=True),
                                action.astype(np.float32, copy=True)))
                leg = np.clip(default[:12] + .35 * action[:12], *limits[:12, :2].T)
                wheel = np.clip(16. * action[12:], -limits[12:, 2], limits[12:, 2])
                return action, leg, wheel

            row = sim.evaluate(
                model, contract, state, seed, 0., limits=limits, action_clip=6., wheel_scale=16.,
                dynamics_stress=stress, terrain="stairs", wheel_angle_period=4 * np.pi,
                observation_suffix=evaluation.gait_clock, native_planner=planner,
            )
            report.append({
                "seed": seed, "passed": bool(row["passed"]), "switched": switched,
                "top_support_s": float(row["top_support_s"]),
                "peak_impact_n": float(row["peak_impact_n"]),
            })
            if row["passed"]:
                obs, act = map(np.stack, zip(*samples))
                observations.append(obs)
                actions.append(act)
                episode_seeds.append(np.full(len(obs), seed, dtype=np.int64))
    if not observations:
        raise RuntimeError("Hybrid teacher did not produce a successful trajectory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        observations=np.concatenate(observations),
        actions=np.concatenate(actions),
        seed=np.concatenate(episode_seeds),
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "successful_episodes": len(observations), "frames": sum(map(len, observations)),
        "seeds": [row["seed"] for row in report if row["passed"]],
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()

