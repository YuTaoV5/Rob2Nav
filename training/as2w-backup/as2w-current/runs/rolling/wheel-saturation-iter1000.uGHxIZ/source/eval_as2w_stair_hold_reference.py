"""Compare hold controllers from identical supported stair approach states."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from as2w_pitch_hold_adapter import CommandAdapter
from eval_as2w_pitch_hold import supported_wheel_levels
from train_recovery_student import make_actor, numpy_actor
from train_stair_low_speed_course import OmniEnvironment


def trial(env, actor, actions, seed, rise, stage, mode):
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "stair_rise": rise, "ascent_speed": .10})
    env.stair_hold_enabled = False
    for tick, action in enumerate(actions):
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            return None
        if supported_wheel_levels(env).min() >= stage:
            break
    else:
        return None

    adapter = CommandAdapter(wheel_hold_gain=3., enable_yaw=False, hold_pitch_gain=2. if mode == "pitch" else 0.)
    initial_pose = np.clip((env.data.qpos[env.qa[:12]] - env.defaults[:12]) / .35, -6., 6.)
    env.target.fill(0.)
    obs = env.observe()
    positions = [float(env.data.qpos[0])]
    residual = []
    peak = 0.
    for _ in range(round(1.5 / env.dt)):
        action = actor(obs)
        if mode != "raw":
            action = adapter.apply(obs, action)
            if mode == "measured-pose":
                action[:12] = initial_pose
        obs, _, terminated, truncated, _ = env.step(action)
        positions.append(float(env.data.qpos[0]))
        residual.append(float(np.linalg.norm(env.data.qvel[:2])))
        peak = max(peak, env.last_stair_impact)
        if terminated or truncated:
            break
    complete = len(residual) == round(1.5 / env.dt) and not (terminated or truncated)
    late_speed = float(np.quantile(residual[25:], .95)) if complete else None
    drift = max(positions) - min(positions)
    backslide = max(0., positions[0] - min(positions))
    passed = complete and late_speed <= .05 and drift <= .06 and backslide <= .03
    resume = False
    if complete:
        rear_on_release = env.levels[2:].min()
        env.target[0] = env.ascent_speed
        obs = env.observe()
        for _ in range(round(4. / env.dt)):
            obs, _, terminated, truncated, _ = env.step(actor(obs))
            peak = max(peak, env.last_stair_impact)
            if terminated or truncated:
                break
            if env.levels[2:].min() > rear_on_release:
                resume = True
                break
    return dict(seed=seed, rise_m=rise, stage=stage, mode=mode, approach_tick=tick + 1,
                hold_completed=complete, held=bool(passed), resumed=resume, drift_m=drift,
                backslide_m=backslide, residual_p95_m_s=late_speed, peak_impact_n=peak)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("references", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
    actor = numpy_actor(make_actor(state))
    env = OmniEnvironment(args.assets, 0, low_speed_course=True, stop_training=True)
    env.apply_motion_prior = False
    rows = []
    with np.load(args.references) as archive:
        seeds, rises, actions = (archive[key] for key in ("seed", "rise_m", "actions"))
    for seed in np.unique(seeds):
        selection = seeds == seed
        for stage in (2, 3):
            for mode in ("raw", "wheel", "pitch", "measured-pose"):
                row = trial(env, actor, actions[selection], int(seed), float(rises[selection][0]), stage, mode)
                if row is not None:
                    rows.append(row)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
