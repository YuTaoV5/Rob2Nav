"""Measure the first riser with paired full and short approaches at 0.10 m/s."""
import argparse
import json
from pathlib import Path

import numpy as np
import mujoco
import torch

from as2w_history import History
from train_as2w_platform_diagnostic import OmniEnvironment
from train_recovery_student import make_actor, numpy_actor


def trial(env, actor, use_history, seed, start_gap):
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "ascent_speed": .10})
    env.apply_motion_prior = False
    front = sorted(env.wheel_index, key=env.wheel_index.get)[:2]
    first = env.model.geom("stair_1")
    edge = float(first.pos[0] - first.size[0])
    top = float(first.pos[2] + first.size[2])
    radius = .086
    if start_gap is not None:
        env.data.qpos[0] += edge - radius - start_gap - env.data.xpos[front, 0].max()
        mujoco.mj_forward(env.model, env.data)
        env.start_x = env.max_x = float(env.data.qpos[0])
        obs = env.observe()

    history = History()
    closest = float("inf")
    max_near_clearance = None
    first_approach_s = None
    wheel_side_frames = 0
    peak_impact = 0.
    for tick in range(round(14. / env.dt)):
        obs, _, terminated, truncated, _ = env.step(actor(history.push(obs) if use_history else obs))
        xyz = env.data.xpos[front]
        gaps = edge - xyz[:, 0] - radius
        closest = min(closest, float(gaps.min()))
        near = gaps <= .03
        if near.any():
            if first_approach_s is None:
                first_approach_s = (tick + 1) * env.dt
            clearance = float((xyz[near, 2] - radius - top).max())
            max_near_clearance = clearance if max_near_clearance is None else max(max_near_clearance, clearance)
        side = any(
            first.id in contact.geom and abs(contact.frame[2]) < .7
            and any(env.model.geom_bodyid[g] in front for g in contact.geom)
            for contact in env.data.contact
        )
        wheel_side_frames += int(side)
        peak_impact = max(peak_impact, env.last_stair_impact)
        if min(env.levels[:2]) >= 1 or terminated or truncated:
            break
    return dict(seed=seed, start_gap_m=start_gap, approach_s=first_approach_s,
                min_gap_m=closest, max_near_clearance_m=max_near_clearance,
                side_contact_frames=wheel_side_frames, front_levels=env.levels[:2].tolist(),
                peak_impact_n=peak_impact, terminated=terminated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = []
    for checkpoint in args.checkpoints:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
        module = make_actor(state)
        use_history = next(module.parameters()).shape[1] > 60
        actor = numpy_actor(module)
        for rise in (.06, .07, .08):
            env = OmniEnvironment(args.assets, 0, stair_rise=rise, low_speed_course=True)
            for seed in (27001, 27004, 36001, 36004):
                for gap in (None, .03):
                    row = trial(env, actor, use_history, seed, gap)
                    rows.append(dict(row, rise_m=rise, checkpoint=str(checkpoint)))
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
