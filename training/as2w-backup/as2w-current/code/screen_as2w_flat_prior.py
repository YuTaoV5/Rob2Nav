"""Measure whether the lateral/yaw motion prior still helps the unified policy."""
import argparse
import json
from pathlib import Path
from statistics import mean

import torch

from eval_as2w_low_speed_course import run_flat
from train_recovery_student import make_actor, numpy_actor
from train_stair_low_speed_course import OmniEnvironment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, weights_only=True, map_location="cpu")["model_state_dict"]
    actor = numpy_actor(make_actor(state))
    commands = {"lateral": [(0., .10, 0.), (0., -.10, 0.)],
                "diagonal_turn": [(.071, .071, .20), (-.071, -.071, -.20)],
                "yaw": [(0., 0., .30), (0., 0., -.30)]}
    rows = []
    for enabled in (True, False):
        env = OmniEnvironment(args.assets, 1, stop_training=True)
        env.apply_motion_prior = enabled
        for label, targets in commands.items():
            group = [run_flat(env, actor, seed, target) for seed in range(36001, 36005) for target in targets]
            rows.extend(dict(row, prior=enabled, scenario=label) for row in group)
            track = [r["tracking_m_s"] for r in group if r["tracking_m_s"] is not None]
            yaw = [r["yaw_error_rad_s"] for r in group if r["yaw_error_rad_s"] is not None]
            print(enabled, label, round(mean(track), 3) if track else None,
                  round(mean(yaw), 3) if yaw else None, sum(r["held"] for r in group),
                  sum(r["contact_frames"] for r in group), sum(r["terminated"] for r in group), flush=True)
    args.output.write_text(json.dumps(rows, indent=2, default=lambda value: value.item()) + "\n")


if __name__ == "__main__":
    main()
