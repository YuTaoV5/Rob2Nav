"""Collect verified flat rolling, turning and release trajectories for student imitation."""
import argparse
import json
from pathlib import Path

import numpy as np

from train_stair_low_speed_course import OmniEnvironment


COMMANDS = ((.10, 0., 0.), (-.10, 0., 0.), (0., 0., .30), (0., 0., -.30))
WHEEL_Y = np.array((.15, -.15, .15, -.15))


def trial(env, seed, command):
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "target": command})
    env.next_command_tick = 10**9
    observations, actions, moving, hold = [], [], [], []
    wheel_reference = None
    contacts = 0
    for tick in range(250):
        if tick == 125:
            env.target_goal.fill(0.)
        rotation = env.data.xmat[env.base].reshape(3, 3)
        vx, vy, _ = rotation.T @ env.data.qvel[:3]
        yaw = float(env.data.qvel[5])
        target_x, _, target_yaw = env.command
        wheel_q = env.data.qpos[env.qa[12:]]
        if not np.any(env.command) and tick >= 125:
            if wheel_reference is None:
                wheel_reference = wheel_q.copy()
            wheel_speed = np.clip(-3. * (wheel_q - wheel_reference), -2., 2.)
        else:
            drive = target_x + .60 * (target_x - vx)
            turn = .75 * target_yaw + .65 * (target_yaw - yaw)
            wheel_speed = np.clip((drive - WHEEL_Y * turn) / .086, -5., 5.)
        action = np.r_[np.zeros(12), wheel_speed / 16.].astype(np.float32)
        observations.append(obs[:58].copy())
        actions.append(action)
        obs, _, terminated, _, _ = env.step(action)
        contacts += any(
            env.ground in c.geom and
            env.model.geom_bodyid[c.geom[1] if c.geom[0] == env.ground else c.geom[0]] not in env.wheel_index
            for c in env.data.contact
        )
        if 40 <= tick < 125:
            moving.append((vx, vy, yaw))
        if tick >= 175:
            hold.append((np.hypot(vx, vy), abs(yaw)))
        if terminated:
            break
    record = dict(seed=seed, command=command, frames=tick + 1, nonwheel_frames=contacts,
                  passed=False, signed_response=None, residual_m_s=None, residual_rad_s=None)
    if moving:
        velocities = np.asarray(moving)
        axis = 0 if command[0] else 2
        record["signed_response"] = float(np.mean(velocities[:, axis]) * np.sign(command[axis]))
    if len(hold) == 75:
        residual_speed, residual_yaw = np.quantile(hold, .95, axis=0)
        record["residual_m_s"] = float(residual_speed)
        record["residual_rad_s"] = float(residual_yaw)
        xy_error = np.linalg.norm(velocities[:, :2] - command[:2], axis=1).mean()
        yaw_error = np.abs(velocities[:, 2] - command[2]).mean()
        record["tracking_m_s"] = float(xy_error)
        record["yaw_error_rad_s"] = float(yaw_error)
        record["passed"] = bool(not contacts and not terminated and xy_error <= .05 and yaw_error <= .14
                                and residual_speed <= .05 and residual_yaw <= .10)
    return record, np.asarray(observations), np.asarray(actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = OmniEnvironment(args.assets, 1, stop_training=True)
    env.apply_motion_prior = False
    report = {}
    obs, act, ids = [], [], []
    for split, start, count in (("train", 73001, 16), ("heldout", 74001, 8)):
        rows = []
        for index, command in enumerate(COMMANDS):
            for seed in range(start + index * 100, start + index * 100 + count):
                row, observations, actions = trial(env, seed, command)
                rows.append(row)
                if split == "train" and row["passed"]:
                    obs.append(observations)
                    act.append(actions)
                    ids.append(np.full(len(actions), seed))
        report[split] = rows
        print(split, [sum(r["passed"] for r in rows if r["command"] == command) for command in COMMANDS],
              flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if obs:
        np.savez_compressed(args.output / "teacher-train.npz", observations=np.concatenate(obs),
                            actions=np.concatenate(act), seed=np.concatenate(ids))


if __name__ == "__main__":
    main()
