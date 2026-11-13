"""Generate fresh flat-only stationary and direction-reversal demonstrations."""
import argparse
import json
from pathlib import Path

import numpy as np

from train_stair_low_speed_course import OmniEnvironment


def trial(env, seed, sequence):
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "target": (0., 0., 0.)})
    env.next_command_tick = 10**9
    wheel_ref = None
    observations, actions, rows = [], [], []
    for speed, seconds in sequence:
        env.target_goal = np.array((speed, 0., 0.))
        planar, signed, positions = [], [], []
        contacts = 0
        phase_frames = round(seconds / env.dt)
        for tick in range(phase_frames):
            rotation = env.data.xmat[env.base].reshape(3, 3)
            vx = float(rotation[:, 0] @ env.data.qvel[:3])
            desired = env.command[0]
            wheel_q = env.data.qpos[env.qa[12:]]
            if desired == 0. and speed == 0.:
                if wheel_ref is None:
                    wheel_ref = wheel_q.copy()
                wheel_speed = np.clip(-3. * (wheel_q - wheel_ref), -2., 2.)
            else:
                wheel_ref = None
                drive = desired + .60 * (desired - vx)
                yaw = float(env.data.qvel[5])
                wheel_speed = np.clip((drive + .65 * yaw * np.array((.15, -.15, .15, -.15))) / .086, -5., 5.)
            action = np.r_[np.zeros(12), wheel_speed / 16.].astype(np.float32)
            observations.append(obs[:58].copy())
            actions.append(action)
            obs, _, terminated, _, _ = env.step(action)
            if tick >= (round(1. / env.dt) if speed == 0. else phase_frames // 2):
                rotation = env.data.xmat[env.base].reshape(3, 3)
                v = (rotation.T @ env.data.qvel[:3])[:2]
                planar.append(float(np.linalg.norm(v - (speed, 0.))))
                signed.append(float(v[0] * np.sign(speed)))
                positions.append(env.data.qpos[:2].copy())
            contacts += any(
                env.ground in c.geom and
                env.model.geom_bodyid[c.geom[1] if c.geom[0] == env.ground else c.geom[0]] not in env.wheel_index
                for c in env.data.contact
            )
            if terminated:
                break
        residual = float(np.quantile(planar, .95)) if planar else None
        drift = float(np.max(np.linalg.norm(np.asarray(positions) - positions[0], axis=1))) if positions else None
        response = float(np.mean(signed)) if signed and speed else None
        passed = bool(not terminated and not contacts and (
            residual <= .02 and drift <= .02 if not speed else response >= .07 and residual <= .06
        ))
        rows.append(dict(command_m_s=speed, response_m_s=response, error_p95_m_s=residual,
                         drift_m=drift, passed=passed, terminated=terminated, contact_frames=contacts))
        if terminated:
            break
    return rows, observations, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = OmniEnvironment(args.assets, 1, stop_training=True)
    env.apply_motion_prior = False
    sequences = {
        "static": ((0., 10.),),
        "forward-reverse": ((.10, 2.), (-.10, 2.), (0., 3.)),
        "reverse-forward": ((-.10, 2.), (.10, 2.), (0., 3.)),
    }
    report = {}
    observations, actions, ids = [], [], []
    for split, start, count in (("standard", 88001, 12), ("heldout", 89001, 6)):
        report[split] = {}
        for index, (name, sequence) in enumerate(sequences.items()):
            rows = []
            for seed in range(start + index * 100, start + index * 100 + count):
                phases, obs, act = trial(env, seed, sequence)
                passed = len(phases) == len(sequence) and all(row["passed"] for row in phases)
                rows.append(dict(seed=seed, passed=passed, phases=phases))
                if split == "standard" and passed:
                    observations.extend(obs)
                    actions.extend(act)
                    ids.extend([seed] * len(obs))
            report[split][name] = rows
            print(split, name, sum(row["passed"] for row in rows), len(rows), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if observations:
        np.savez_compressed(args.output / "teacher-train.npz", observations=observations,
                            actions=actions, seed=ids)


if __name__ == "__main__":
    main()
