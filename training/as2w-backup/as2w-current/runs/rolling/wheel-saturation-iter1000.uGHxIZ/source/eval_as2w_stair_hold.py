"""Measure a requested zero-speed hold after first stair contact."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_recovery_student import make_actor, numpy_actor
from train_stair_history_ab_stop_v2 import OmniEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    actor = numpy_actor(make_actor(torch.load(args.checkpoint, weights_only=True, map_location="cpu")["model_state_dict"]))
    rows = []
    for rise in (.06, .08):
        for seed in range(32101, 32117):
            env = OmniEnvironment(args.assets, 0, stair_speed_control=True, stair_rise=rise, stop_training=True)
            obs, _ = env.reset(seed=seed, options={"scene_seed": seed})
            env.stair_hold_enabled = True
            x = []
            speeds = []
            hold_failed = False
            for _ in range(450):
                holding = not np.any(env.command)
                if holding and not x:
                    x.append(float(env.data.qpos[0]))
                obs, _, terminated, truncated, _ = env.step(actor(obs))
                if holding:
                    x.append(float(env.data.qpos[0]))
                    speeds.append(float(np.linalg.norm(env.data.qvel[:2])))
                    hold_failed |= terminated
                if terminated or truncated:
                    break
            rows.append({"seed": seed, "rise_m": rise, "hold_started": bool(speeds),
                         "hold_samples": len(speeds), "hold_terminated": hold_failed,
                         "backslide_m": max(0., x[0] - min(x)) if x else None,
                         "late_hold_speed_p95_m_s": float(np.quantile(speeds[25:], .95)) if len(speeds) > 25 else None,
                         "max_wheel_level": int(env.levels.max())})
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
