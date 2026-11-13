"""Screen whether a faster stair actor transfers to slower wheel targets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from train_recovery_student import make_actor, numpy_actor
from train_stair_low_speed_course import OmniEnvironment


def screen(case):
    assets, ckpt, rise, seed, scale, actor_speed = case
    torch.set_num_threads(1)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"]
    actor = numpy_actor(make_actor(state))
    env = OmniEnvironment(assets, 0, stair_rise=rise, low_speed_course=True)
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "ascent_speed": .10})
    wheels = sorted(env.wheel_index, key=env.wheel_index.get)
    edges = np.asarray([env.model.geom(f"stair_{i}").pos[0] - env.model.geom(f"stair_{i}").size[0]
                        for i in range(1, 6)])
    adaptive = scale < 0.
    peak, speed_sum, elapsed = 0., 0., 0.
    top_x = None
    continued = False
    for tick in range(round(env.stair_seconds / env.dt)):
        frame = obs.copy()
        frame[0] = actor_speed - .4
        action = actor(frame)
        if adaptive:
            # Privileged teacher probe. Keep extra drive local to riser transfer.
            next_edge = edges[np.minimum(env.levels, 4)]
            distance = next_edge - env.data.xpos[wheels, 0] - .086
            near_edge = np.any((env.levels < 5) & (distance < .10) & (distance > -.08))
            speed = float(env.data.qvel[0])
            if min(env.levels) == 5:
                action[:12] = 0.
                action[12:] = .10 / (.086 * 16.)
            elif near_edge:
                action[12:] *= np.clip(1. + 2. * (.10 - speed), .45, -scale)
            else:
                action[12:] *= np.clip(.48 + 2. * (.10 - speed), .30, .80)
        else:
            action[12:] *= scale
        obs, _, terminated, truncated, _ = env.step(action)
        peak = max(peak, env.last_stair_impact)
        speed_sum += abs(float(env.data.qvel[0]) - .10)
        elapsed = (tick + 1) * env.dt
        x = float(env.data.qpos[0])
        if env.top_streak >= round(.5 / env.dt) and top_x is None:
            top_x = x
        if top_x is not None and env.top_streak > 0 and x >= top_x + .25:
            continued = True
            break
        if terminated or truncated:
            break
    return dict(rise_m=rise, seed=seed, wheel_scale=scale, actor_command_m_s=actor_speed,
                levels=env.levels.tolist(), peak_impact_n=peak, top_stable=top_x is not None,
                continued=continued, elapsed_s=elapsed, terminated=terminated,
                forward_error_m_s=speed_sum / (tick + 1),
                passed=bool(continued and not terminated and peak <= 90.))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--adaptive", action="store_true",
                        help="Compare two riser-local drive boosts with full-scale control")
    args = parser.parse_args()
    settings = ((-1.15, .4), (-1.35, .4), (1., .4)) if args.adaptive else (
        (1., .10), (.4, .4), (.65, .4), (1., .4))
    rises = (.06, .07, .08) if args.adaptive else (.06, .08)
    cases = [(args.assets, args.checkpoint, rise, seed, scale, actor_speed)
             for rise in rises for seed in (27001, 27002, 36001, 36002)
             for scale, actor_speed in settings]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(screen, cases))
    args.output.write_text(json.dumps(rows, indent=2, default=lambda value: value.item()) + "\n")
    for rise in rises:
        for scale, actor_speed in settings:
            group = [r for r in rows if (r["rise_m"], r["wheel_scale"], r["actor_command_m_s"]) ==
                     (rise, scale, actor_speed)]
            print(rise, scale, actor_speed, sum(r["passed"] for r in group),
                  sum(max(r["levels"]) == 5 for r in group),
                  round(float(np.mean([r["forward_error_m_s"] for r in group])), 3), flush=True)


if __name__ == "__main__":
    main()
