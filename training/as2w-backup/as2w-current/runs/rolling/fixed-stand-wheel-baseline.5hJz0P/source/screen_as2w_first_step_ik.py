"""Probe sequential, geometry-assisted wheel transfer onto the first 8 cm step."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from eval_as2w_pitch_hold import supported_wheel_levels
from train_recovery_student import make_actor, numpy_actor
from train_stair_low_speed_course import OmniEnvironment


def trial(env, actor, seed, clearance):
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "ascent_speed": .10})
    wheels = sorted(env.wheel_index, key=env.wheel_index.get)
    riser = env.model.geom("stair_1")
    edge = riser.pos[0] - riser.size[0]
    top = riser.pos[2] + riser.size[2]
    radius = .086
    jac = np.zeros((3, env.model.nv))
    active = None
    phase = "lift"
    attempts = np.zeros(4, dtype=int)
    peak = 0.
    full_support = False
    for tick in range(round(20. / env.dt)):
        action = actor(obs)
        centers = env.data.xpos[wheels]
        if active is not None and (env.levels[active] >= 1 or tick - started >= round(1.5 / env.dt)):
            active = None
        if clearance and active is None:
            eligible = np.flatnonzero((env.levels == 0) & (centers[:, 0] >= edge - radius - .04)
                                      & (centers[:, 0] <= edge + .04) & (attempts < 2))
            if len(eligible):
                active = int(eligible[0])
                attempts[active] += 1
                started = tick
                anchor = centers[active].copy()
                phase = "lift"
        if active is not None:
            wheel = centers[active]
            lifted_z = top + radius + clearance
            if phase == "lift" and wheel[2] >= lifted_z - .015:
                phase = "transfer"
            if phase == "transfer" and wheel[0] >= edge + .065:
                phase = "lower"
            desired = anchor.copy()
            if phase != "lift":
                desired[0] = edge + .11
            desired[2] = top + radius if phase == "lower" else lifted_z
            joints = np.array((active, 4 + active, 8 + active))
            dofs = np.asarray(env.va)[joints]
            mujoco.mj_jacBody(env.model, env.data, jac, None, wheels[active])
            leg_jac = jac[:, dofs]
            dq = leg_jac.T @ np.linalg.solve(leg_jac @ leg_jac.T + .0025 * np.eye(3), desired - wheel)
            q = env.data.qpos[np.asarray(env.qa)[joints]] + np.clip(1.6 * dq, -.35, .35)
            action[joints] = (q - env.defaults[joints]) / .35
            action[12:] = np.clip(action[12:], -.08, .12)
            action[12 + active] = 0.
        obs, _, terminated, truncated, _ = env.step(np.clip(action, -6., 6.))
        peak = max(peak, env.last_stair_impact)
        full_support = bool(env.levels.min() >= 1 and supported_wheel_levels(env).min() >= 1)
        if terminated or truncated or full_support:
            break
    return dict(seed=seed, clearance_m=clearance, levels=env.levels.tolist(), full_support=full_support,
                attempts=attempts.tolist(), terminated=terminated, peak_impact_n=peak, frames=tick + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    actor = numpy_actor(make_actor(state))
    env = OmniEnvironment(args.assets, 0, stair_rise=.08, low_speed_course=True)
    env.apply_motion_prior = False
    rows = []
    for clearance in (0., .025, .045):
        group = [trial(env, actor, seed, clearance) for seed in (*range(85001, 85007), *range(86001, 86007))]
        rows.extend(group)
        print(clearance, "front", sum(min(row["levels"][:2]) >= 1 for row in group),
              "rear", sum(min(row["levels"][2:]) >= 1 for row in group),
              "four", sum(row["full_support"] for row in group), flush=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
