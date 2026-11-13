"""Evaluate standing and move-to-zero transitions in the flat simulator."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_recovery_student import make_actor, numpy_actor
from train_stair_history_ab_stop import OmniEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    actor = numpy_actor(make_actor(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]))
    scenarios = {"stand": (0., 0., 0.), "forward": (.10, 0., 0.), "reverse": (-.10, 0., 0.),
                 "lateral": (0., .10, 0.), "yaw": (0., 0., .30)}
    rows = []
    for seed in range(42001, 42009):
        for label, command in scenarios.items():
            env = OmniEnvironment(args.assets, 1, low_speed_hind_scale=0., stop_training=True)
            obs, _ = env.reset(seed=seed, options={"target": command})
            env.next_command_tick = 10**9
            speed, yaw = [], []
            for tick in range(300):
                if tick == 125:
                    env.target_goal.fill(0.)
                obs, _, terminated, truncated, _ = env.step(actor(obs))
                if tick >= 175 and not np.any(env.command):
                    speed.append(float(np.linalg.norm(env.data.qvel[:2])))
                    yaw.append(abs(float(env.data.qvel[5])))
                if terminated or truncated:
                    break
            rows.append({"seed": seed, "scenario": label, "terminated": terminated,
                         "hold_samples": len(speed), "speed_p95_m_s": float(np.quantile(speed, .95)) if speed else None,
                         "yaw_p95_rad_s": float(np.quantile(yaw, .95)) if yaw else None})
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
