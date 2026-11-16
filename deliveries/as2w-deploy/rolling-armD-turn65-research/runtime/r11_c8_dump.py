#!/usr/bin/env python3
"""C8 minimal replay: capture ONE real Isaac rollout frame by frame.

Checklist v3 C8 asks for a representative fixed input + initial state together with, per
frame, the model's raw output, the final control target and the next frame's history /
phase / wheel reference, plus a recomputation script and a report of the max error, the
worst field and the first divergence.

This script produces the capture side:
  * a fixed initial state        -> env.seed(SEED) + env.reset(), 2 s settle at zero command
  * a fixed command sequence     -> a single frozen case (default arc_right_0.25)
  * per frame, 50 Hz (policy rate): obs[65], raw actor output[16], final wheel velocity
    target[4] (after the action manager and clip), measured wheel dq[4], body yaw rate,
    and the NEXT frame's obs
The parity side is done by r11_c8_verify.py, which runs policy.onnx on every recorded obs
and reports where it first diverges from the recorded raw output.

No gates, no pass/fail: this is a capture for replay, and the gate verdict for the same
checkpoint already exists in the acceptance evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sim import SimulationCfg

VX_CAP, YAW_CAP = .25, .15
WHEEL_KD, ACTION_CLIP = 6.0, 6.0
DT = 0.02
SETTLE_S = 2.0

CASES = {
    "arc_right_0.25": (0.25, 0.0, -0.05, 12.0),
    "arc_left_0.25": (0.25, 0.0, +0.05, 12.0),
    "turn_right_0.05": (0.0, 0.0, -0.05, 12.0),
    "hold": (0.0, 0.0, 0.0, 10.0),
}


def build(args, physics):
    from isaaclab.managers import RewardTermCfg
    from as2w_isaac_flat_env_all import (
        RslRlVecEnvWrapper, UnitreeGo2FlatPPORunnerCfg,
        check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg)
    from as2w_isaac_robust_env import RobustEnvironment, TakeoverCommands, robust_config
    from as2w_isaac_speed_range import wheel_support_error
    from as2w_training_safety import apply_safety, rolling_knee_margin_cost

    class EvalCommands(TakeoverCommands):
        def _resample_command(self, env_ids):
            pass

    cfg = robust_config(SimpleNamespace(
        **vars(args), profile="pose-delay", direction_focus=True,
        command_stage="all", zero_memory=True, zero_focus=True), physics)
    cmd = cfg.commands.base_velocity
    cmd.class_type = EvalCommands
    cmd.resampling_time_range = (1e6, 1e6)
    cmd.ranges.lin_vel_x = (-VX_CAP, VX_CAP)
    cmd.ranges.lin_vel_y = (0., 0.)
    cmd.ranges.ang_vel_z = (-YAW_CAP, YAW_CAP)
    cfg.episode_length_s = 32.
    cfg.scene.robot.actuators["foot"].damping = WHEEL_KD
    cfg.rewards.track_lin_vel_xy_exp.weight = 1.5
    cfg.rewards.track_ang_vel_z_exp.weight = 1.5
    cfg.rewards.track_ang_vel_z_exp.params["std"] = .08
    cfg.rewards.rolling_wheel_support = RewardTermCfg(func=wheel_support_error, weight=-.65)
    cfg.rewards.rolling_knee_margin = RewardTermCfg(
        func=rolling_knee_margin_cost, weight=-3., params={"margin_m": .10})
    apply_safety(cfg, args.urdf, perturb=False, surface_guard=True)
    env = RobustEnvironment(cfg)
    return (env, RslRlVecEnvWrapper, UnitreeGo2FlatPPORunnerCfg,
            check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--case", default="arc_right_0.25", choices=sorted(CASES))
    ap.add_argument("--seed", type=int, default=7393)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=1)
    add_launcher_args(ap)
    args = ap.parse_args()
    for a in ("usd", "urdf", "ckpt"):
        setattr(args, a, getattr(args, a).resolve(strict=True))
    args.out.mkdir(parents=True, exist_ok=True)
    args.physics = "isaacsim_physx"
    vx, vy, wz, dur = CASES[args.case]

    with launch_simulation(SimulationCfg(device=args.device, dt=.005), args) as physics:
        (env, Wrapper, PPOCfg, check_v, create_runner, handle_dep) = build(args, physics)
        try:
            wrapped = Wrapper(env, clip_actions=ACTION_CLIP)
            robot = env.scene["robot"]
            wj = env.as2w_joint_ids[12:]
            cmdterm = env.command_manager.get_term("base_velocity")
            wterm = env.action_manager.get_term("wheel_vel")

            ppo = PPOCfg(); ppo.seed, ppo.device = args.seed, args.device
            ppo.max_iterations = 1
            ppo = handle_dep(ppo, check_v())
            runner = create_runner(wrapped, ppo, log_dir=None)
            runner.load(str(args.ckpt), map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)
            print("C8_CKPT", args.ckpt, "iter", runner.current_learning_iteration, flush=True)

            env.seed(args.seed); env.reset()
            cmdterm.vel_command_b[:] = 0.0
            obs = wrapped.get_observations()
            for _ in range(int(SETTLE_S / DT)):
                with torch.inference_mode():
                    a = policy(obs)
                obs, _rew, _don, _info = wrapped.step(a)

            rows = []
            n = int(dur / DT)
            for k in range(n):
                cmdterm.vel_command_b[:] = torch.tensor(
                    [[vx, vy, wz]], dtype=torch.float32, device=env.device)
                o = obs["policy"][0].detach().cpu().numpy().astype(np.float32)
                with torch.inference_mode():
                    a = policy(obs)
                raw = a[0].detach().cpu().numpy().astype(np.float32)
                obs, _rew, _don, _info = wrapped.step(a)
                tgt = wterm.delayed()[0].detach().cpu().numpy().astype(np.float32)
                dq = robot.data.joint_vel.torch[0, wj].detach().cpu().numpy().astype(np.float32)
                rows.append({
                    "frame": k,
                    "t_s": round(k * DT, 4),
                    "cmd": [float(vx), float(vy), float(wz)],
                    "obs": [float(v) for v in o],
                    "raw_actor_output": [float(v) for v in raw],
                    "wheel_target": [float(v) for v in tgt],
                    "wheel_dq_measured": [float(v) for v in dq],
                    "yaw_rate": float(robot.data.root_ang_vel_b.torch[0, 2]),
                    "upright": float(-robot.data.projected_gravity_b.torch[0][2]),
                    "next_obs": [float(v) for v in
                                 obs["policy"][0].detach().cpu().numpy().astype(np.float32)],
                })
            cap = {
                "case": args.case,
                "command": {"vx": vx, "vy": vy, "wz": wz, "duration_s": dur},
                "seed": args.seed,
                "initial_state": {"protocol": "env.seed(seed) + env.reset(), then %.1f s settle at zero command" % SETTLE_S},
                "policy_period_s": DT,
                "action_clip": ACTION_CLIP,
                "wheel_kd": WHEEL_KD,
                "obs_dim": 65, "action_dim": 16,
                "n_frames": len(rows),
                "note": "obs is the 65-dim policy observation; raw_actor_output is the network output BEFORE the action manager; wheel_target is the final velocity target after the manager and clip",
                "frames": rows,
            }
            (args.out / ("c8_capture_%s.json" % args.case)).write_text(json.dumps(cap))
            print("C8_CAPTURED", args.case, len(rows), "frames", flush=True)
            print("C8_DONE", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
