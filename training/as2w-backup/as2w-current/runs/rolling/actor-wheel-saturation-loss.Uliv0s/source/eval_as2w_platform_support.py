"""Aligned flat and low-speed five-step screens for a shared actor."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from as2w_history import History
from train_recovery_student import make_actor, numpy_actor
from train_stair_low_speed_course import OmniEnvironment


def body_velocity(env):
    rotation = env.data.xmat[env.base].reshape(3, 3)
    return (rotation.T @ env.data.qvel[:3])[:2]


def run_stairs(env, infer, seed, stop):
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "ascent_speed": .10})
    env.stair_hold_enabled = stop
    top_x = None
    longest_support_frames = 0
    continued = False
    peak = 0.
    moving, hold, hold_x = [], [], []
    for _ in range(round(env.stair_seconds / env.dt)):
        command = env.command.copy()
        obs, _, terminated, truncated, _ = env.step(infer(obs))
        peak = max(peak, env.last_stair_impact)
        x = float(env.data.qpos[0])
        if np.any(command):
            moving.append(float(np.linalg.norm(body_velocity(env) - command[:2])))
        else:
            hold.append(float(np.linalg.norm(body_velocity(env))))
            hold_x.append(x)
        if env.top_streak >= round(.5 / env.dt) and top_x is None:
            top_x = x
        longest_support_frames = max(longest_support_frames, env.top_streak)
        if top_x is not None and env.top_streak > 0 and x >= top_x + .25 and command[0] > 0.:
            continued = True
        if terminated or truncated:
            break
    full_hold = len(hold) == round(1.5 / env.dt)
    residual = float(np.quantile(hold[25:], .95)) if full_hold else None
    drift = max(hold_x) - min(hold_x) if hold_x else None
    backslide = max(0., hold_x[0] - min(hold_x)) if hold_x else None
    held = full_hold and residual <= .05 and drift <= .06 and backslide <= .03
    return dict(seed=seed, interrupted=stop, levels=env.levels.tolist(), peak_impact_n=peak,
                terminated=terminated, top_stable=top_x is not None, platform_continued=continued,
                longest_platform_support_s=longest_support_frames * env.dt,
                passed=bool(continued and not terminated and peak <= 90. and (not stop or held)),
                moving_error_m_s=float(np.mean(moving)), hold_frames=len(hold),
                hold_speed_p95_m_s=residual, hold_drift_m=drift, backslide_m=backslide, held=held)


def run_flat(env, infer, seed, target):
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "target": target})
    env.next_command_tick = 10**9
    tracking, yaw, hind, pitch, hold, hold_yaw = [], [], [], [], [], []
    touches = 0
    for tick in range(300):
        if tick == 150:
            env.target_goal.fill(0.)
        command = env.command.copy()
        obs, _, terminated, truncated, _ = env.step(infer(obs))
        velocity = body_velocity(env)
        if 25 <= tick < 150:
            tracking.append(float(np.linalg.norm(velocity - command[:2])))
            yaw.append(abs(float(env.data.qvel[5] - command[2])))
        if tick >= 200:
            hold.append(float(np.linalg.norm(velocity)))
            hold_yaw.append(abs(float(env.data.qvel[5])))
        q = env.data.qpos[env.qa] - env.defaults
        hind.extend(np.abs(q[[2, 3, 6, 7, 10, 11]]))
        pitch.append(abs(float(np.degrees(np.arcsin(np.clip(env.data.xmat[env.base, 6], -1, 1))))))
        touches += any(env.ground in c.geom and
                       env.model.geom_bodyid[c.geom[1] if c.geom[0] == env.ground else c.geom[0]] not in env.wheel_index
                       for c in env.data.contact)
        if terminated or truncated:
            break
    return dict(seed=seed, target=target, terminated=terminated, contact_frames=touches,
                tracking_m_s=float(np.mean(tracking)) if tracking else None,
                yaw_error_rad_s=float(np.mean(yaw)) if yaw else None,
                hind_p95_rad=float(np.quantile(hind, .95)), pitch_p95_deg=float(np.quantile(pitch, .95)),
                held=len(hold) == 100 and np.quantile(hold, .95) <= .05 and np.quantile(hold_yaw, .95) <= .10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--no-motion-prior", action="store_true")
    parser.add_argument("--wheel-hold-gain", type=float, help="Enable the experimental command adapter")
    parser.add_argument("--hold-only", action="store_true", help="Keep raw action when the command is nonzero")
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    actor = numpy_actor(make_actor(state))
    uses_history = state["actor.0.weight"].shape[1] == History.size

    def new_inference():
        history = History()
        adapter = None
        if args.wheel_hold_gain is not None:
            from as2w_command_adapter import CommandAdapter
            adapter = CommandAdapter(args.wheel_hold_gain, enable_yaw=not args.hold_only)

        def infer(obs):
            action = actor(history.push(obs) if uses_history else obs)
            return action if adapter is None else adapter.apply(obs, action)

        return infer

    report = {}
    targets = ((.10, 0., 0.), (-.10, 0., 0.), (0., .10, 0.), (0., -.10, 0.),
               (.071, .071, .20), (-.071, -.071, -.20), (0., 0., .30), (0., 0., 0.))
    for label, start in (("standard", 27001), ("heldout", 36001)):
        seeds = range(start, start + args.count)
        flat = OmniEnvironment(args.assets, 1, stop_training=True)
        flat.apply_motion_prior = not args.no_motion_prior and args.wheel_hold_gain is None
        stairs = {}
        for rise in (.06, .07, .08):
            env = OmniEnvironment(args.assets, 0, stair_rise=rise, stop_training=True, low_speed_course=True)
            stairs[str(rise)] = [run_stairs(env, new_inference(), seed, stop)
                                 for seed in seeds for stop in (False, True)]
        report[label] = dict(stairs=stairs, flat=[run_flat(flat, new_inference(), seed, target)
                                                 for seed in seeds for target in targets])
        print(label, {rise: sum(row["passed"] for row in rows) for rise, rows in stairs.items()}, flush=True)
    args.output.write_text(json.dumps(report, indent=2, default=lambda value: value.item()) + "\n")


if __name__ == "__main__":
    main()
