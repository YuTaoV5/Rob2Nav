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
    peak, speed_sum, elapsed = 0., 0., 0.
    top_x = None
    continued = False
    for tick in range(round(env.stair_seconds / env.dt)):
        frame = obs.copy()
        frame[0] = actor_speed - .4
        action = actor(frame)
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
    args = parser.parse_args()
    cases = [(args.assets, args.checkpoint, rise, seed, scale, actor_speed)
             for rise in (.06, .08) for seed in (27001, 27002, 36001, 36002)
             for scale, actor_speed in ((1., .10), (.4, .4), (.65, .4), (1., .4))]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(screen, cases))
    args.output.write_text(json.dumps(rows, indent=2, default=lambda value: value.item()) + "\n")
    for rise in (.06, .08):
        for scale, actor_speed in ((1., .10), (.4, .4), (.65, .4), (1., .4)):
            group = [r for r in rows if (r["rise_m"], r["wheel_scale"], r["actor_command_m_s"]) ==
                     (rise, scale, actor_speed)]
            print(rise, scale, actor_speed, sum(r["passed"] for r in group),
                  sum(max(r["levels"]) == 5 for r in group),
                  round(float(np.mean([r["forward_error_m_s"] for r in group])), 3), flush=True)


if __name__ == "__main__":
    main()
