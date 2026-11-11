"""Explain fifth-step contact without a stable four-wheel platform finish."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_recovery_student import make_actor, numpy_actor
from train_stair_low_speed_course import OmniEnvironment, sim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=27004)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, weights_only=True, map_location="cpu")["model_state_dict"]
    actor = numpy_actor(make_actor(state))
    env = OmniEnvironment(args.assets, 0, stair_rise=.06, stop_training=True, low_speed_course=True)
    obs, _ = env.reset(seed=args.seed, options={"scene_seed": args.seed, "ascent_speed": .10})
    env.stair_hold_enabled = True
    top = env.model.geom("stair_5")
    top_id = top.id
    surfaces = set(env.surfaces)
    top_front, top_back = float(top.pos[0] - top.size[0]), float(top.pos[0] + top.size[0])
    wheel_ids = sorted(env.wheel_index, key=env.wheel_index.get)
    rows = []
    for tick in range(round(env.stair_seconds / env.dt)):
        obs, _, terminated, truncated, _ = env.step(actor(obs))
        if env.levels.max() >= 4:
            loads = np.zeros(4)
            impact_body = None
            strongest = 0.
            for index, contact in enumerate(env.data.contact):
                g1, g2 = contact.geom
                if (g1 in surfaces) != (g2 in surfaces):
                    robot_geom = g2 if g1 in surfaces else g1
                    body = env.model.geom_bodyid[robot_geom]
                    if body not in env.wheel_index:
                        sim.mj.mj_contactForce(env.model, env.data, index, env.force)
                        if env.force[0] > strongest:
                            strongest = float(env.force[0])
                            impact_body = env.model.body(body).name
                if top_id not in (g1, g2) or abs(contact.frame[2]) < .7:
                    continue
                robot = g2 if g1 == top_id else g1
                wheel = env.wheel_index.get(env.model.geom_bodyid[robot])
                if wheel is not None:
                    sim.mj.mj_contactForce(env.model, env.data, index, env.force)
                    loads[wheel] += env.force[0]
            rows.append(dict(time_s=tick * env.dt, levels=env.levels.tolist(),
                             top_loads_n=loads.tolist(), body_x=float(env.data.qpos[0]),
                             body_y=float(env.data.qpos[1]), upright=float(env.data.xmat[env.base, 8]),
                             wheel_x=env.data.xpos[wheel_ids, 0].tolist(),
                             wheel_y=env.data.xpos[wheel_ids, 1].tolist(),
                             nonwheel_body=impact_body,
                             peak_nonwheel_n=float(env.last_stair_impact)))
        if terminated or truncated:
            break
    args.output.write_text(json.dumps(dict(top_front_m=top_front, top_back_m=top_back,
                                           rows=rows), indent=2) + "\n")
    if rows:
        simultaneous = [r for r in rows if min(r["top_loads_n"]) >= 8.]
        stable = [r for r in simultaneous if r["upright"] >= .94 and r["peak_nonwheel_n"] < 1.]
        print("final levels", rows[-1]["levels"], "four-load frames", len(simultaneous),
              "upright low-impact frames", len(stable), "last body x/y",
              [round(rows[-1][key], 3) for key in ("body_x", "body_y")])


if __name__ == "__main__":
    main()
