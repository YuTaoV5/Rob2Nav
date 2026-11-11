"""Measure steady drift on flat ground with a constant zero joystick command."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from as2w_history import History
from as2w_zero_memory import ZeroWheelMemory
from as2w_command_adapter import CommandAdapter
from train_recovery_student import make_actor, numpy_actor
from train_stair_low_speed_course import OmniEnvironment


def episode(env, actor, history_enabled, seed, hold_gain=None, nominal_legs=False, zero_memory=False):
    obs, _ = env.reset(seed=seed, options={"scene_seed": seed, "target": (0., 0., 0.)})
    env.next_command_tick = 10**9
    history = History()
    memory = ZeroWheelMemory() if zero_memory else None
    hold = CommandAdapter(hold_gain, enable_yaw=False) if hold_gain is not None else None
    xy, yaw, speed = [], [], []
    contacts = 0
    for tick in range(round(10. / env.dt)):
        features = history.push(obs) if history_enabled else obs
        action = actor(memory.push(features) if memory is not None else features)
        if hold is not None:
            action = hold.apply(obs, action)
        if nominal_legs:
            action[:12] = 0.
        obs, _, terminated, _, _ = env.step(action)
        if tick >= round(1. / env.dt):
            rotation = env.data.xmat[env.base].reshape(3, 3)
            xy.append(env.data.qpos[:2].copy())
            yaw.append(float(np.arctan2(rotation[1, 0], rotation[0, 0])))
            speed.append(float(np.linalg.norm(env.data.qvel[:2])))
        contacts += any(
            env.ground in c.geom and
            env.model.geom_bodyid[c.geom[1] if c.geom[0] == env.ground else c.geom[0]] not in env.wheel_index
            for c in env.data.contact
        )
        if terminated:
            break
    drift = float(np.max(np.linalg.norm(np.asarray(xy) - xy[0], axis=1))) if xy else None
    heading = float(np.rad2deg(np.max(np.abs(np.unwrap(yaw) - yaw[0])))) if yaw else None
    residual = float(np.quantile(speed, .95)) if speed else None
    return dict(seed=seed, frames=tick + 1, terminated=terminated, drift_m=drift,
                yaw_drift_deg=heading, residual_p95_m_s=residual, nonwheel_frames=contacts,
                passed=bool(not terminated and not contacts and drift <= .02 and heading <= 3. and residual <= .02))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--hold-wheel-gain", type=float)
    parser.add_argument("--nominal-legs-on-zero", action="store_true")
    args = parser.parse_args()
    if args.hold_wheel_gain is not None and (not np.isfinite(args.hold_wheel_gain) or args.hold_wheel_gain < 0.):
        parser.error("Wheel hold gain must be finite and nonnegative")
    torch.set_num_threads(1)
    env = OmniEnvironment(args.assets, 1, stop_training=True)
    env.apply_motion_prior = False
    report = {}
    for item in args.checkpoints:
        label, path = item.split("=", 1)
        state = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
        actor = numpy_actor(make_actor(state))
        uses_history = state["actor.0.weight"].shape[1] == History.size
        zero_memory = state["actor.0.weight"].shape[1] == ZeroWheelMemory.size
        report[label] = {}
        for split, start in (("standard", 27001), ("heldout", 37001)):
            rows = [episode(env, actor, uses_history, seed, args.hold_wheel_gain, args.nominal_legs_on_zero, zero_memory)
                    for seed in range(start, start + 4)]
            report[label][split] = rows
            print(label, split, sum(r["passed"] for r in rows), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
