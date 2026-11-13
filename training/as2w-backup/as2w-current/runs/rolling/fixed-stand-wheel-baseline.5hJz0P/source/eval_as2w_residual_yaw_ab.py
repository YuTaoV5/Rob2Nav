"""Screen a simulation-only flat teacher from low-speed rolling through 2 m/s."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from as2w_history import History
from as2w_longitudinal_prior import longitudinal_prior
from train_recovery_student import make_actor, numpy_actor
from train_stair_low_speed_course import OmniEnvironment


SPEEDS = (.10, .25, .50, 1., 1.5, 2., 0.)


def trial(env, seed, deceleration, student=None, uses_history=False, velocity_source="simulator", hip_trim=0.):
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "target": (0., 0., 0.)})
    env.next_command_tick = 10**9
    rows, observations, actions = [], [], []
    wheel_reference = None
    history = History()
    for target in SPEEDS:
        env.target_goal = np.array((target, 0., 0.))
        speeds, lateral, yaws, hind = [], [], [], []
        signed_rear = []
        touches = 0
        phase_start = env.data.qpos[:2].copy()
        phase_frames = 200 if target == 0. and deceleration else 100
        for tick in range(phase_frames):
            rotation = env.data.xmat[env.base].reshape(3, 3)
            vx, _, _ = rotation.T @ env.data.qvel[:3]
            yaw = float(env.data.qvel[5])
            requested = env.command[0]
            env.flat_command_alpha = .12
            limit_rate = target == 0. if student is None else max(target, requested) > .15
            if deceleration and limit_rate:
                env.flat_command_alpha = min(.12, deceleration * env.dt / max(abs(target - requested), .001))
            wheel_q = env.data.qpos[env.qa[12:]]
            if student is not None:
                action = student(history.push(obs) if uses_history else obs)
            elif target == 0. and requested == 0.:
                if wheel_reference is None:
                    wheel_reference = wheel_q.copy()
                wheel_speed = np.clip(-3. * (wheel_q - wheel_reference), -2., 2.)
                action = np.r_[np.zeros(12), wheel_speed / 16.].astype(np.float32)
            else:
                if velocity_source == "encoder":
                    vx = .086 * env.data.qvel[env.va[12:]].mean()
                drive = requested if velocity_source == "none" else requested + .60 * (requested - vx)
                turn = -.65 * yaw
                wheel_speed = np.clip((drive - np.array((.15, -.15, .15, -.15)) * turn) / .086, -30., 30.)
                action = np.r_[np.zeros(12), wheel_speed / 16.].astype(np.float32)
                trim = hip_trim * np.clip((requested - .20) / 1.80, 0., 1.) / .35
                action[2:4] = (-trim, trim)
            observations.append(obs[:58].copy())
            actions.append(action)
            obs, _, terminated, _, _ = env.step(action)
            rotation = env.data.xmat[env.base].reshape(3, 3)
            vx, vy, _ = rotation.T @ env.data.qvel[:3]
            rear_error = (env.data.qpos[env.qa] - env.defaults)[[2, 3, 6, 7, 10, 11]]
            if tick >= phase_frames - 50:
                speeds.append(float(vx))
                lateral.append(float(vy))
                yaws.append(abs(float(env.data.qvel[5])))
                signed_rear.append(rear_error)
            hind.append(np.abs(rear_error))
            touches += any(
                env.ground in c.geom and
                env.model.geom_bodyid[c.geom[1] if c.geom[0] == env.ground else c.geom[0]] not in env.wheel_index
                for c in env.data.contact
            )
            if terminated:
                break
        tracking = float(np.mean(np.abs(np.asarray(speeds) - target))) if speeds else None
        residual = float(np.quantile(np.hypot(speeds, lateral), .95)) if speeds and target == 0. else None
        passed = bool(not terminated and not touches and len(speeds) == 50 and
                      tracking <= max(.05, .10 * target) and np.quantile(yaws, .95) <= .10 and
                      (residual is None or residual <= .05))
        rows.append(dict(seed=seed, command_m_s=target, frames=tick + 1, passed=passed,
                         vx_m_s=float(np.mean(speeds)) if speeds else None, error_m_s=tracking,
                         lateral_m_s=float(np.mean(np.abs(lateral))) if lateral else None,
                         yaw_p95_rad_s=float(np.quantile(yaws, .95)) if yaws else None,
                         rear_p95_rad=float(np.quantile(hind, .95)), contact_frames=touches,
                         signed_rear_median_rad=np.median(signed_rear, axis=0).tolist() if signed_rear else None,
                         residual_p95_m_s=residual, terminated=terminated,
                         displacement_m=float(np.linalg.norm(env.data.qpos[:2] - phase_start))))
        if terminated:
            break
    return rows, observations, actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--deceleration", type=float, default=0., help="Limit the filtered release ramp in m/s²")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate a blind student instead of the teacher")
    parser.add_argument("--longitudinal-residual", action="store_true",
                        help="Apply the command-based reference to a residual checkpoint")
    parser.add_argument("--suppress-fast-yaw-residual", action="store_true",
                        help="For diagnosis, leave straight high-speed wheel yaw to the reference")
    parser.add_argument("--velocity-source", choices=("simulator", "encoder", "none"), default="simulator",
                        help="Longitudinal feedback for the teacher only")
    parser.add_argument("--hip-trim", type=float, default=0.,
                        help="Inward rear hip target offset in radians at 2 m/s for the teacher")
    parser.add_argument("--seed-offset", type=int, default=0, help="Shift both evaluation seed ranges")
    args = parser.parse_args()
    if not np.isfinite(args.deceleration) or args.deceleration < 0.:
        parser.error("Deceleration must be finite and nonnegative")
    if not np.isfinite(args.hip_trim) or not 0. <= args.hip_trim <= .5:
        parser.error("Hip trim must be in [0, 0.5] radians")
    if args.longitudinal_residual and not args.checkpoint:
        parser.error("The residual action reference requires a student checkpoint")
    if args.suppress_fast_yaw_residual and not args.longitudinal_residual:
        parser.error("Yaw residual suppression requires a residual checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    env = OmniEnvironment(args.assets, 1, flat_command_max=2., stop_training=True)
    env.apply_motion_prior = False
    student = None
    uses_history = False
    if args.checkpoint:
        torch.set_num_threads(1)
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
        uses_history = state["actor.0.weight"].shape[1] == History.size
        raw_actor = numpy_actor(make_actor(state))
        if args.longitudinal_residual:
            yaw_pattern = np.array((-1., 1., -1., 1.), dtype=np.float32)

            def student(features):
                residual = raw_actor(features)
                if args.suppress_fast_yaw_residual:
                    vx = features[0] + .4
                    # Preserve all low-speed commands and the longitudinal wheel residual.
                    weight = np.clip((vx - .15) / .10, 0., 1.)
                    weight *= np.clip(1. - abs(features[1]) / .08 - abs(features[2]) / .20, 0., 1.)
                    residual[12:] -= weight * (residual[12:] @ yaw_pattern) * yaw_pattern / 4.
                return np.clip(residual + longitudinal_prior(features), -6., 6.)
        else:
            student = raw_actor
    report = {}
    observations, actions, seeds = [], [], []
    for split, start, count in (("standard", 78001, 8), ("heldout", 79001, 8)):
        report[split] = []
        for seed in range(start + args.seed_offset, start + args.seed_offset + count):
            rows, obs, act = trial(env, seed, args.deceleration, student, uses_history,
                                   args.velocity_source, args.hip_trim)
            report[split].extend(rows)
            if student is None and split == "standard" and len(rows) == len(SPEEDS) and all(row["passed"] for row in rows):
                observations.extend(obs)
                actions.extend(act)
                seeds.extend([seed] * len(obs))
        print(split, [(speed, sum(r["passed"] for r in report[split] if r["command_m_s"] == speed))
                      for speed in SPEEDS], flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if observations:
        np.savez_compressed(args.output / "teacher-train.npz", observations=observations,
                            actions=actions, seed=seeds)


if __name__ == "__main__":
    main()
