"""Simulate signed yaw and lateral command primitives without measured base velocity."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from search_flat_omni_teacher import StepTeacher
from train_recovery_student import make_actor, numpy_actor
from train_stair_low_speed_course import OmniEnvironment


def evaluate(case):
    assets, ckpt, family, setting, seed, sign = case
    torch.set_num_threads(1)
    env = OmniEnvironment(assets, 1, stop_training=True)
    env.apply_motion_prior = False
    state = torch.load(ckpt, weights_only=True, map_location="cpu")["model_state_dict"]
    actor = numpy_actor(make_actor(state))
    teacher = StepTeacher(env.contract).policy(setting) if family == "lateral" else None
    cmd = (0., .10 * sign, 0.) if teacher is not None else (0., 0., .30 * sign)
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "target": cmd})
    env.next_command_tick = 10**9
    signed, cross, hold, hold_yaw = [], [], [], []
    touches = 0
    hold_reference = None
    for tick in range(300):
        if tick == 150:
            env.target_goal.fill(0.)
        a = actor(obs)
        if teacher is not None:
            gait = teacher(obs)
            strength = min(abs(env.command[1]) / .08, 1.)
            a = (1. - strength) * a + strength * gait
        elif family in ("yaw", "yaw_hold"):
            feed, feedback, legs = setting[:3]
            yaw = float(env.command[2])
            strength = min(abs(yaw) / .18, 1.)
            turn = np.clip((feed * yaw + feedback * (yaw - obs[5])) / (.086 * 16.), -.8, .8)
            a[:12] *= 1. - legs * strength
            a[12:] = (1. - strength) * a[12:] + strength * turn * np.array([-1., 1., -1., 1.])
            if family == "yaw_hold" and tick >= 150 and not np.any(env.command):
                if hold_reference is None:
                    hold_reference = obs[22:26].copy()
                    held_legs = obs[42:54].copy()
                delta = (obs[22:26] - hold_reference + 2. * np.pi) % (4. * np.pi) - 2. * np.pi
                a[:12] = held_legs
                a[12:] = np.clip(-setting[3] * delta, -2., 2.) / 16.
        obs, _, terminated, truncated, _ = env.step(a)
        rotation = env.data.xmat[env.base].reshape(3, 3)
        v = (rotation.T @ env.data.qvel[:3])[:2]
        w = float(env.data.qvel[5])
        if 40 <= tick < 150:
            signed.append(sign * (v[1] if family.startswith("lateral") else w))
            cross.append(np.linalg.norm(v) if family.startswith("yaw") else abs(v[0]))
        if tick >= 200:
            hold.append(float(np.linalg.norm(v)))
            hold_yaw.append(abs(w))
        touches += any(env.ground in c.geom and
                       env.model.geom_bodyid[c.geom[1] if c.geom[0] == env.ground else c.geom[0]] not in env.wheel_index
                       for c in env.data.contact)
        if terminated or truncated:
            break
    return dict(family=family, setting=setting, seed=seed, direction=sign,
                signed=float(np.mean(signed)) if signed else None,
                drift_m_s=float(np.mean(cross)) if cross else None, nonwheel_frames=touches,
                terminated=terminated, held=len(hold) == 100 and
                np.quantile(hold, .95) <= .05 and np.quantile(hold_yaw, .95) <= .10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--yaw-hold", action="store_true")
    args = parser.parse_args()
    settings = [("yaw_raw", ()), ("lateral_raw", ())]
    settings += [("yaw", (f, p, legs)) for f in (.20, .45, .75) for p in (.25, .55) for legs in (.4, 1.)]
    settings += [("lateral", (.62, .23, height, stride, .18, .55, .05, .65, .015))
                 for height in (.055, .10) for stride in (-2.5, -1.25, 1.25, 2.5)]
    if args.yaw_hold:
        settings = [("yaw_raw", ()), ("yaw", (.45, .55, 1.))]
        settings += [("yaw_hold", (feed, .55, 1., gain)) for feed in (.20, .32, .45) for gain in (0., 3., 8.)]
    cases = [(args.assets, args.checkpoint, family, cfg, seed, sign)
             for family, cfg in settings for seed in (58101, 58102, 58201, 58202) for sign in (-1, 1)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(evaluate, cases))
    args.output.write_text(json.dumps(rows, indent=2, default=lambda x: x.item()) + "\n")
    for family, cfg in settings:
        group = [r for r in rows if r["family"] == family and r["setting"] == cfg]
        print(family, cfg,
              [round(mean(r["signed"] for r in group if r["direction"] == sign and r["signed"] is not None), 3)
               for sign in (-1, 1)],
              "hold", sum(r["held"] for r in group), "touch", sum(r["nonwheel_frames"] for r in group),
              "term", sum(r["terminated"] for r in group), flush=True)


if __name__ == "__main__":
    main()
