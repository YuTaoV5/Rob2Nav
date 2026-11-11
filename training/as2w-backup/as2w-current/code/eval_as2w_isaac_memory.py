"""Evaluate multi-command low-speed joystick sequences with a shared actor."""
import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from as2w_history import History
from as2w_zero_memory import ZeroWheelMemory
from train_recovery_student import make_actor, numpy_actor
from train_stair_low_speed_course import OmniEnvironment


COMMANDS = (
    ("forward", (.10, 0., 0.)),
    ("left", (0., .10, 0.)),
    ("diagonal", (-.071, .071, -.20)),
    ("yaw", (0., 0., .30)),
    ("reverse", (-.10, 0., 0.)),
    ("stop", (0., 0., 0.)),
)
SAGITTAL_COMMANDS = (
    ("forward", (.10, 0., 0.)),
    ("reverse", (-.10, 0., 0.)),
    ("stop", (0., 0., 0.)),
)
TURN_COMMANDS = (
    ("forward", (.10, 0., 0.)),
    ("yaw-left", (0., 0., .30)),
    ("reverse", (-.10, 0., 0.)),
    ("yaw-right", (0., 0., -.30)),
    ("stop", (0., 0., 0.)),
)
ALL_FLAT_COMMANDS = TURN_COMMANDS[:-1] + (
    ("left", (0., .10, 0.)),
    ("right", (0., -.10, 0.)),
    ("diagonal-fl", (.071, .071, 0.)),
    ("diagonal-fr", (.071, -.071, 0.)),
    ("diagonal-rl", (-.071, .071, 0.)),
    ("diagonal-rr", (-.071, -.071, 0.)),
    ("stop", (0., 0., 0.)),
)


def episode(env, actor, seed, history_enabled, wheel_hold_gain=None, enable_yaw=True,
            mask_wheel_angle=False, use_raw_action_history=False, privileged=False, commands=COMMANDS,
            zero_memory=False):
    mirrored = seed % 2 == 0
    history = History()
    memory = ZeroWheelMemory() if zero_memory else None
    adapter = None
    last_raw_action = np.zeros(16, dtype=np.float32)
    if wheel_hold_gain is not None:
        from as2w_command_adapter import CommandAdapter
        adapter = CommandAdapter(wheel_hold_gain, enable_yaw=enable_yaw)
    reset_options = {"scene_seed": seed, "target": (0., 0., 0.)}
    if privileged:
        from as2w_privileged_critic import PrivilegedCriticEnv
        teacher_env = PrivilegedCriticEnv(env)
        obs, _ = teacher_env.reset(seed=seed, options=reset_options)
        obs = obs[:60]
    else:
        obs, _ = env.reset(seed=seed, options=reset_options)
    env.next_command_tick = 10**9
    rows = []
    for label, target in commands:
        cmd = np.asarray(target) * ([1, -1, -1] if mirrored else [1, 1, 1])
        env.target_goal = cmd
        velocities, yaw_rates, hind, pitch = [], [], [], []
        contacts = 0
        duration = round(2. / env.dt)
        for tick in range(duration):
            actor_obs = obs.copy()
            if mask_wheel_angle:
                actor_obs[22:26] = 0.
            if use_raw_action_history:
                actor_obs[42:58] = last_raw_action
            if privileged:
                actor_obs = teacher_env.observation(actor_obs)
            if history_enabled:
                actor_obs = history.push(actor_obs)
            if memory is not None:
                actor_obs = memory.push(actor_obs)
            action = actor(actor_obs)
            last_raw_action = action.copy()
            if adapter is not None:
                action = adapter.apply(obs, action)
            # This explicit command sequence runs beyond the randomized training episode.
            obs, _, terminated, _, _ = env.step(action)
            rotation = env.data.xmat[env.base].reshape(3, 3)
            if tick >= round(.5 / env.dt):
                velocities.append((rotation.T @ env.data.qvel[:3])[:2])
                yaw_rates.append(float(env.data.qvel[5]))
            q = env.data.qpos[env.qa] - env.defaults
            hind.append(np.abs(q[[2, 3, 6, 7, 10, 11]]))
            pitch.append(abs(float(np.degrees(np.arcsin(np.clip(rotation[2, 0], -1., 1.))))))
            contacts += any(
                env.ground in c.geom and
                env.model.geom_bodyid[c.geom[1] if c.geom[0] == env.ground else c.geom[0]] not in env.wheel_index
                for c in env.data.contact
            )
            if terminated:
                break
        vel = np.asarray(velocities).reshape(-1, 2)
        yaw = np.asarray(yaw_rates)
        signed_progress = None
        if len(vel) and np.any(cmd[:2]):
            signed_progress = float(np.mean(vel @ (cmd[:2] / np.linalg.norm(cmd[:2]))))
        rows.append(dict(
            seed=seed, segment=label, command=cmd.tolist(), frames=tick + 1,
            terminated=terminated, tracking_m_s=float(np.mean(np.linalg.norm(vel - cmd[:2], axis=1))) if len(vel) else None,
            yaw_error_rad_s=float(np.mean(np.abs(yaw - cmd[2]))) if len(yaw) else None,
            signed_progress_m_s=signed_progress, signed_yaw_rad_s=float(np.mean(yaw) * np.sign(cmd[2])) if len(yaw) else None,
            held=bool(label == "stop" and tick + 1 == duration and not terminated and
                      np.quantile(np.linalg.norm(vel, axis=1), .95) <= .05 and np.quantile(np.abs(yaw), .95) <= .10),
            hind_p95_rad=float(np.quantile(hind, .95)), pitch_p95_deg=float(np.quantile(pitch, .95)),
            hind_joint_p95_rad=np.quantile(hind, .95, axis=0).tolist(),
            nonwheel_frames=contacts,
        ))
        if terminated:
            break
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoints", nargs="+",
                        help="label:prior|raw|hold3|hold8|neutral3|mask3|rawhistory3|maskraw|mirror|mirror-phase:checkpoint")
    commands = parser.add_mutually_exclusive_group()
    commands.add_argument("--sagittal", action="store_true", help="Evaluate forward, reverse and stop only")
    commands.add_argument("--turn", action="store_true", help="Evaluate longitudinal motion and both yaw directions")
    commands.add_argument("--all-flat", action="store_true", help="Evaluate both signs of each axis and all diagonals")
    args = parser.parse_args()
    if args.sagittal:
        sequence = SAGITTAL_COMMANDS
    elif args.turn:
        sequence = TURN_COMMANDS
    elif args.all_flat:
        sequence = ALL_FLAT_COMMANDS
    else:
        sequence = COMMANDS
    torch.set_num_threads(1)
    report = {}
    for item in args.checkpoints:
        label, mode, checkpoint = item.split(":", 2)
        if mode not in {"prior", "raw", "hold3", "hold8", "neutral3", "mask3", "rawhistory3",
                        "maskraw", "mirror", "mirror-phase"}:
            parser.error("Unknown evaluation controller mode")
        state = torch.load(checkpoint, weights_only=True, map_location="cpu")["model_state_dict"]
        input_size = state["actor.0.weight"].shape[1]
        if input_size not in (60, 81, History.size, ZeroWheelMemory.size):
            parser.error("Unexpected checkpoint input shape")
        if input_size == 81 and mode != "raw":
            parser.error("Use raw mode for the simulation-only teacher")
        actor = numpy_actor(make_actor(state))
        if mode.startswith("mirror"):
            if input_size != 60:
                parser.error("Mirrored inference requires the canonical 60-input actor")
            from as2w_mirror_actor import mirrored_actor
            actor = mirrored_actor(actor, flip_clock=mode == "mirror-phase")
        env = OmniEnvironment(args.assets, 1, stop_training=True)
        env.apply_motion_prior = mode == "prior"
        for split, seeds in (("standard", range(27001, 27005)), ("heldout", range(37001, 37005))):
            gain = float(mode[-1]) if mode.endswith(("3", "8")) else None
            rows = [r for seed in seeds
                    for r in episode(env, actor, seed, input_size == History.size,
                                     gain, enable_yaw=mode != "neutral3",
                                     mask_wheel_angle=mode.startswith("mask"),
                                     use_raw_action_history=mode == "rawhistory3", privileged=input_size == 81,
                                     commands=sequence, zero_memory=input_size == ZeroWheelMemory.size)]
            report.setdefault(label, {})[split] = rows
            move = [r for r in rows if r["segment"] != "stop" and r["tracking_m_s"] is not None]
            yaw = [r for r in move if r["segment"].startswith("yaw")]
            print(label, split, "track", round(mean(r["tracking_m_s"] for r in move), 3) if move else None,
                  "yaw", round(mean(r["yaw_error_rad_s"] for r in yaw), 3) if yaw else None,
                  "held", sum(r["held"] for r in rows), "touch", sum(r["nonwheel_frames"] for r in rows),
                  "terminated", sum(r["terminated"] for r in rows), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
