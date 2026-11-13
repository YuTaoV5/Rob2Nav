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
    assets, ckpt, rise, seed, scale, actor_speed, collect, steering, rear, lift, hold_gain = case
    torch.set_num_threads(1)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)["model_state_dict"]
    actor = numpy_actor(make_actor(state))
    env = OmniEnvironment(assets, 0, stair_rise=rise, low_speed_course=True)
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "ascent_speed": .10})
    env.stair_hold_enabled = hold_gain >= 0.
    wheels = sorted(env.wheel_index, key=env.wheel_index.get)
    edges = np.asarray([env.model.geom(f"stair_{i}").pos[0] - env.model.geom(f"stair_{i}").size[0]
                        for i in range(1, 6)])
    adaptive = scale < 0.
    peak, speed_sum, elapsed = 0., 0., 0.
    top_x = None
    continued = False
    observations, actions = [], []
    hold_position = None
    hold_speeds, hold_x = [], []
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
                # Encourage rear transfer when the front pair is already one step above.
                if min(env.levels[:2]) > min(env.levels[2:]):
                    action[6:8] += .55 * rear
                    action[10:12] -= .65 * rear
                    action[14:16] += .20 * rear
                if lift:
                    # Residual from the faster 8 cm assist, applied only at a riser.
                    action[:12] += lift * np.repeat(
                        [-.1519, -.0795, -.676, 1.3946, -.9985, -1.384], 2)
                    action[12:] += lift * np.repeat([.6339, 1.0246], 2)
            else:
                action[12:] *= np.clip(.48 + 2. * (.10 - speed), .30, .80)
            if steering:
                rotation = env.data.xmat[env.base].reshape(3, 3)
                yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
                correction = steering * (env.data.qpos[1] + .45 * yaw)
                action[12:] += np.clip(correction, -.22, .22) * np.array([1., -1., 1., -1.])
        else:
            action[12:] *= scale
        holding = not np.any(env.command)
        if holding:
            wheel_q = env.data.qpos[env.qa[12:]]
            if hold_position is None:
                hold_position = wheel_q.copy()
                held_legs = env.previous[:12].copy()
                hold_start = float(env.data.qpos[0])
            action[:12] = held_legs
            action[12:] = np.clip(-hold_gain * (wheel_q - hold_position), -2., 2.) / 16.
        np.clip(action, -6., 6., out=action)
        if collect:
            observations.append(obs.copy())
            actions.append(action.copy())
        obs, _, terminated, truncated, _ = env.step(action)
        peak = max(peak, env.last_stair_impact)
        speed_sum += abs(float(env.data.qvel[0]) - .10)
        elapsed = (tick + 1) * env.dt
        x = float(env.data.qpos[0])
        if holding:
            hold_speeds.append(float(np.linalg.norm(env.data.qvel[:2])))
            hold_x.append(x)
        if env.top_streak >= round(.5 / env.dt) and top_x is None:
            top_x = x
        if top_x is not None and env.top_streak > 0 and x >= top_x + .25:
            continued = True
            break
        if terminated or truncated:
            break
    complete_hold = len(hold_speeds) == round(1.5 / env.dt)
    residual = float(np.quantile(hold_speeds[25:], .95)) if complete_hold else None
    backslide = max(0., hold_start - min(hold_x)) if hold_x else None
    held = bool(complete_hold and residual <= .05 and backslide <= .03
                and max([hold_start, *hold_x]) - min([hold_start, *hold_x]) <= .06)
    row = dict(rise_m=rise, seed=seed, wheel_scale=scale, actor_command_m_s=actor_speed,
                steering_gain=steering, rear_gain=rear, lift_gain=lift,
                hold_gain=hold_gain, hold_frames=len(hold_speeds), hold_speed_p95_m_s=residual,
                backslide_m=backslide, held=held,
                levels=env.levels.tolist(), peak_impact_n=peak, top_stable=top_x is not None,
                continued=continued, elapsed_s=elapsed, terminated=terminated,
                forward_error_m_s=speed_sum / (tick + 1),
                passed=bool(continued and not terminated and peak <= 90. and (hold_gain < 0. or held)))
    if collect and row["passed"]:
        return row, np.asarray(observations), np.asarray(actions)
    return row, None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--adaptive", action="store_true",
                        help="Compare two riser-local drive boosts with full-scale control")
    parser.add_argument("--collect", type=Path, help="Save passing demonstrations on disjoint training seeds")
    parser.add_argument("--collect-count", type=int, default=24)
    parser.add_argument("--eight-search", action="store_true", help="Screen rear transfer and heading correction at 8 cm")
    parser.add_argument("--eight-residual", action="store_true", help="Screen the faster 8 cm assist residual under low-speed control")
    parser.add_argument("--hold-search", action="store_true", help="Screen wheel position hold with the last supporting leg targets")
    parser.add_argument("--collect-hold", type=Path, help="Archive successful interrupted courses on new training seeds")
    args = parser.parse_args()
    settings = ((-1.15, .4), (-1.35, .4), (1., .4)) if args.adaptive else (
        (1., .10), (.4, .4), (.65, .4), (1., .4))
    rises = (.06, .07, .08) if args.adaptive else (.06, .08)
    if args.collect:
        rises, settings = (.06, .07, .08), ((-1.35, .4),)
    cases = [(args.assets, args.checkpoint, rise, seed, scale, actor_speed, bool(args.collect), 0., 0., 0., -1.)
             for index, rise in enumerate(rises)
             for seed in (range(51001 + 100 * index, 51001 + 100 * index + args.collect_count)
                          if args.collect else (27001, 27002, 36001, 36002))
             for scale, actor_speed in settings]
    if args.eight_search:
        cases = [(args.assets, args.checkpoint, .08, seed, -cap, .4, False, steer, rear, 0., -1.)
                 for seed in range(53001, 53009) for cap in (1.35, 1.65)
                 for steer in (0., .50, 1.0) for rear in (0., .5, 1.)]
    if args.eight_residual:
        cases = [(args.assets, args.checkpoint, .08, seed, -cap, .4, False, steer, 0., lift, -1.)
                 for seed in range(55001, 55009) for cap in (1.35, 1.65)
                 for steer in (-.50, 0., .50) for lift in (0., .35, .70, 1.)]
    if args.hold_search:
        cases = [(args.assets, args.checkpoint, rise, seed, -1.35, .4, False, 0., 0., 0., gain)
                 for height_index, rise in enumerate((.06, .07, .08))
                 for seed in range(59001 + 100 * height_index, 59009 + 100 * height_index)
                 for gain in (0., 3., 8.)]
    if args.collect_hold:
        cases = [(args.assets, args.checkpoint, rise, seed, -1.35, .4, True, 0., 0., 0., 8.)
                 for height_index, rise in enumerate((.06, .07, .08))
                 for seed in range(60001 + 100 * height_index, 60001 + 100 * height_index + args.collect_count)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(screen, cases))
    rows = [row for row, _, _ in results]
    args.output.write_text(json.dumps(rows, indent=2, default=lambda value: value.item()) + "\n")
    if args.hold_search or args.collect_hold:
        for rise in (.06, .07, .08):
            for gain in ((8.,) if args.collect_hold else (0., 3., 8.)):
                group = [r for r in rows if r["rise_m"] == rise and r["hold_gain"] == gain]
                print(rise, gain, "triggered", sum(r["hold_frames"] > 0 for r in group),
                      "held", sum(r["held"] for r in group), "passed", sum(r["passed"] for r in group), flush=True)
        if not args.collect_hold:
            return
    if args.eight_search or args.eight_residual:
        def parameters(row):
            return tuple(row[key] for key in ("wheel_scale", "steering_gain", "rear_gain", "lift_gain"))

        for config in sorted({parameters(r) for r in rows}):
            group = [r for r in rows if parameters(r) == config]
            print(config, "passed", sum(r["passed"] for r in group), "fifth",
                  sum(max(r["levels"]) == 5 for r in group), "over90",
                  sum(r["peak_impact_n"] > 90. for r in group), flush=True)
        return
    archive = args.collect or args.collect_hold
    if archive:
        passed = [(r, obs, action) for r, obs, action in results if obs is not None]
        if passed:
            np.savez_compressed(archive, observations=np.concatenate([obs[:, :58] for _, obs, _ in passed]),
                                actions=np.concatenate([action for _, _, action in passed]),
                                seed=np.concatenate([np.full(len(obs), r["seed"]) for r, obs, _ in passed]),
                                rise_m=np.concatenate([np.full(len(obs), r["rise_m"]) for r, obs, _ in passed]))
    for rise in rises:
        for scale, actor_speed in settings:
            group = [r for r in rows if (r["rise_m"], r["wheel_scale"], r["actor_command_m_s"]) ==
                     (rise, scale, actor_speed)]
            print(rise, scale, actor_speed, sum(r["passed"] for r in group),
                  sum(max(r["levels"]) == 5 for r in group),
                  round(float(np.mean([r["forward_error_m_s"] for r in group])), 3), flush=True)


if __name__ == "__main__":
    main()
