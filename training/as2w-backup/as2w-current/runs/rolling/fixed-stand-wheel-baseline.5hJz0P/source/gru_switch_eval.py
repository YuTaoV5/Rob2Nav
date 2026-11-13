"""Evaluate recurrent velocity-free stair-top detection with a latched stabilizer switch."""

import argparse
from collections import Counter
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
    parser.add_argument("--threshold", type=float, default=.80)
    parser.add_argument("--streak", type=int, default=3)
    parser.add_argument("--max-trigger-time", type=float, default=float("inf"))
    parser.add_argument("--wheel-action", type=float, default=.10)
    parser.add_argument("--stair-rise", type=float, default=.06)
    parser.add_argument("--seed-start", type=int, default=27001)
    parser.add_argument("--count", type=int, default=16)
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
    rows = []
    with torch.inference_mode():
        for seed in range(args.seed_start, args.seed_start + args.count):
            model = sim.load_flat_model(True, True, "stairs", seed, stair_rise=args.stair_rise)
            stress = sim.sample_dynamics_stress(seed, model.nbody - 1, 16)
            stair = model.geom("stair_5")
            fifth_leading = float(model.geom_pos[stair.id, 0] - model.geom_size[stair.id, 0])
            hidden = None
            consecutive = 0
            switched = False
            trigger = None

            def planner(data, obs, elapsed, advance):
                nonlocal hidden, consecutive, switched, trigger
                logits, hidden = detector(torch.as_tensor(obs, dtype=torch.float32)[None, None], hidden)
                probability = float(torch.sigmoid(logits)[0, 0])
                consecutive = consecutive + 1 if probability >= args.threshold else 0
                if not switched and elapsed <= args.max_trigger_time and consecutive >= args.streak:
                    switched = True
                    trigger = {
                        "time_s": float(elapsed), "probability": probability,
                        "x_after_fifth_leading_m": float(data.qpos[0] - fifth_leading),
                    }
                action = (np.r_[np.zeros(12), np.full(4, args.wheel_action)]
                          if switched else np.clip(actor(obs), -6., 6.))
                leg = np.clip(default[:12] + .35 * action[:12], *limits[:12, :2].T)
                wheel = np.clip(16. * action[12:], -limits[12:, 2], limits[12:, 2])
                return action, leg, wheel

            row = sim.evaluate(
                model, contract, state, seed, 0., limits=limits, action_clip=6., wheel_scale=16.,
                dynamics_stress=stress, terrain="stairs", wheel_angle_period=4 * np.pi,
                observation_suffix=evaluation.gait_clock, native_planner=planner,
            )
            rows.append({
                "seed": seed, "passed": bool(row["passed"]), "switched": bool(switched),
                "trigger": trigger, "visited_levels": [int(level) for level in row["visited_levels"]],
                "top_support_s": float(row["top_support_s"]),
                "peak_impact_n": float(row["peak_impact_n"]),
            })
    highest = Counter(max(row["visited_levels"], default=0) for row in rows)
    report = {
        "threshold": args.threshold, "streak": args.streak, "wheel_action": args.wheel_action,
        "passed": [sum(row["passed"] for row in rows), len(rows)],
        "switched": sum(row["switched"] for row in rows),
        "false_early_switches": sum(
            row["trigger"] is not None and row["trigger"]["x_after_fifth_leading_m"] < .35
            for row in rows
        ),
        "impact_over_90n": sum(row["peak_impact_n"] > 90. for row in rows),
        "highest_levels": {str(level): int(count) for level, count in highest.items()},
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    main()

