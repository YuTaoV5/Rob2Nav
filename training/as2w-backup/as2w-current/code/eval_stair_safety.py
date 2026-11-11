"""Evaluate one 60-input stair actor on fixed six-centimetre stair seeds."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_omni_stairs as evaluation
import mujoco_screen as sim


def summarize(rows):
    impacts = np.asarray([row["peak_impact_n"] for row in rows])
    speed_p95 = np.asarray([row["forward_speed_p95_m_s"] for row in rows])
    return {
        "passed": [sum(row["passed"] for row in rows), len(rows)],
        "highest_levels": dict(Counter(max(row["visited_levels"], default=0) for row in rows)),
        "progress_m": {
            "p25": float(np.percentile([row["progress_m"] for row in rows], 25)),
            "median": float(np.median([row["progress_m"] for row in rows])),
            "p75": float(np.percentile([row["progress_m"] for row in rows], 75)),
        },
        "impact_n": {
            "p50": float(np.median(impacts)),
            "p95": float(np.percentile(impacts, 95)),
            "max": float(impacts.max()),
        },
        "over_90n_episodes": int(np.count_nonzero(impacts > 90.)),
        "speed_p95_episode_m_s": {
            "median": float(np.median(speed_p95)),
            "max": float(speed_p95.max()),
        },
        "overspeed_step_fraction_mean": float(np.mean([row["overspeed_fraction"] for row in rows])),
        "abs_y_p95_m_median": float(np.median([row["abs_y_p95_m"] for row in rows])),
        "stall_episodes": int(sum(row["progress_m"] < .12 for row in rows)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stair-rise", type=float, default=.06)
    parser.add_argument("--standard-only", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)
    sim.ROOT = args.results.resolve()
    contract = json.loads((sim.ROOT / "fall-baseline/fall-recovery-contract.json").read_text())
    limits = sim.load_target_limits(contract)
    state = torch.load(args.checkpoint, weights_only=True, map_location="cpu")["model_state_dict"]
    if state["actor.0.weight"].shape[1] != 60:
        raise ValueError("Expected a 60-input stair actor")

    splits = {"standard": range(27001, 27017)}
    if not args.standard_only:
        splits["heldout"] = range(31001, 31017)
    by_split = {}
    rows = []
    with torch.inference_mode():
        for split, seeds in splits.items():
            split_rows = []
            for seed in seeds:
                model = sim.load_flat_model(True, True, "stairs", seed, stair_rise=args.stair_rise)
                stress = sim.sample_dynamics_stress(seed, model.nbody - 1, 16)
                base = model.body("base_link").id
                rng = np.random.default_rng(seed)
                rng.uniform(.55, .85)
                start_x = float(rng.uniform((-.12, -.15), (.12, .15))[0])
                forward_speed = []
                lateral = []

                def capture(data):
                    forward_speed.append(float(data.xmat[base, [0, 3, 6]] @ data.qvel[:3]))
                    lateral.append(abs(float(data.qpos[1])))

                row = sim.evaluate(
                    model, contract, state, seed, 0., capture=capture, limits=limits,
                    action_clip=6., wheel_scale=16., dynamics_stress=stress, terrain="stairs",
                    wheel_angle_period=4 * np.pi, observation_suffix=evaluation.gait_clock,
                )
                speed = np.asarray(forward_speed)
                row.update(
                    seed=seed, split=split, progress_m=float(row["max_forward_x_m"] - start_x),
                    forward_speed_p95_m_s=float(np.percentile(speed, 95)),
                    overspeed_fraction=float(np.mean(speed > .55)),
                    abs_y_p95_m=float(np.percentile(lateral, 95)),
                )
                split_rows.append(row)
                rows.append(row)
            by_split[split] = summarize(split_rows)

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "stair_rise_m": args.stair_rise,
        **by_split,
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(by_split))


if __name__ == "__main__":
    main()

