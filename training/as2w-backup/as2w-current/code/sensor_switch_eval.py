"""Evaluate a deployable IMU and wheel-encoder stair-top switch."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_omni_stairs as evaluation
import mujoco_screen as sim
from train_recovery_student import make_actor, numpy_actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--odom-rad", type=float, required=True)
    parser.add_argument("--max-pitch", type=float, required=True)
    parser.add_argument("--pitch-below", type=float, required=True)
    parser.add_argument("--min-time", type=float, default=5.5)
    parser.add_argument("--wheel-action", type=float, default=.10)
    parser.add_argument("--stair-rise", type=float, default=.06)
    parser.add_argument("--seed-start", type=int, default=27001)
    parser.add_argument("--count", type=int, default=16)
    args = parser.parse_args()

    torch.set_num_threads(1)
    sim.ROOT = args.results.resolve()
    contract = json.loads((sim.ROOT / "fall-baseline/fall-recovery-contract.json").read_text())
    dt = contract["control_dt"]
    limits = sim.load_target_limits(contract)
    state = torch.load(args.checkpoint, weights_only=True, map_location="cpu")["model_state_dict"]
    actor = numpy_actor(make_actor(state))
    default = np.asarray(contract["default_joint_pos"])
    rows = []
    with torch.inference_mode():
        for seed in range(args.seed_start, args.seed_start + args.count):
            model = sim.load_flat_model(True, True, "stairs", seed, stair_rise=args.stair_rise)
            stress = sim.sample_dynamics_stress(seed, model.nbody - 1, 16)
            stair = model.geom("stair_5")
            fifth_leading = float(model.geom_pos[stair.id, 0] - model.geom_size[stair.id, 0])
            odom = 0.
            maximum_pitch = 0.
            switched = False
            trigger = None

            def planner(data, obs, elapsed, advance):
                nonlocal odom, maximum_pitch, switched, trigger
                odom += max(0., float(np.mean(obs[38:42]))) * dt
                pitch = float(np.arctan2(obs[6], -obs[8]))
                maximum_pitch = max(maximum_pitch, abs(pitch))
                if (not switched and elapsed >= args.min_time and odom >= args.odom_rad
                        and maximum_pitch >= args.max_pitch and pitch <= args.pitch_below):
                    switched = True
                    trigger = {
                        "time_s": float(elapsed), "odom_rad": odom, "pitch_rad": pitch,
                        "max_pitch_rad": maximum_pitch,
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
        "thresholds": {
            "odom_rad": args.odom_rad, "max_pitch": args.max_pitch,
            "pitch_below": args.pitch_below, "min_time": args.min_time,
            "wheel_action": args.wheel_action,
        },
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

