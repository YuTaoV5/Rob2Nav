"""Compare stair entry at the same initial state under two forward commands."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_recovery_student import make_actor, numpy_actor
from train_stair_low_speed_course import OmniEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = []
    for checkpoint in args.checkpoints:
        state = torch.load(checkpoint, weights_only=True, map_location="cpu")["model_state_dict"]
        actor = numpy_actor(make_actor(state))
        env = OmniEnvironment(args.assets, 0, stair_rise=.06, low_speed_course=True)
        for seed in (27001, 27002, 36001, 36002):
            for speed in (.10, .40):
                obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "ascent_speed": speed})
                first = env.model.geom("stair_1")
                leading = float(first.pos[0] - first.size[0])
                near = []
                entry = None
                for tick in range(round(env.stair_seconds / env.dt)):
                    obs, _, terminated, truncated, _ = env.step(actor(obs))
                    front = max(env.data.xpos[i, 0] for i, leg in env.wheel_index.items() if leg < 2)
                    if front >= leading - .14:
                        if entry is None:
                            entry = tick * env.dt
                        if env.levels.max() == 0:
                            near.append((env.data.qvel[0], *env.previous[12:14] * 16.,
                                         *env.data.qvel[env.va[12:14]], env.data.xpos[env.base, 2]))
                    if terminated or truncated:
                        break
                medians = np.median(near, axis=0).tolist() if near else None
                rows.append(dict(checkpoint=str(checkpoint), seed=seed, speed=speed, entry_s=entry,
                                 levels=env.levels.tolist(), forward_progress=float(env.max_x - env.start_x),
                                 terminated=terminated, near_frames=len(near),
                                 near_columns=["world_vx", "FL_target_dq", "FR_target_dq",
                                               "FL_dq", "FR_dq", "base_height"],
                                 near_medians=medians))
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
