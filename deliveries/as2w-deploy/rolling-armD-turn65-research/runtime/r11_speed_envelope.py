#!/usr/bin/env python3
"""Speed envelope: how fast can this checkpoint actually roll, and how well does it stop.

The goal requires tiered speed validation at 0.25, 0.6 and 1.0 m/s and the final report has
to state the speed range actually achieved.  Only 0.25 has ever been measured, and unlocking
a higher tier is supposed to depend on results - so this probe MEASURES each tier without
implying that any of them passed the acceptance gate.

Per tier, forward and backward:
  drive  10 s at the tier command  -> vx tracking ratio, four-wheel support, upright
  brake   5 s drive then 5 s at zero -> stop distance, stop time, final drift
Reported, nothing gated.  The verdict column of the acceptance matrix stays authoritative.
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

VX_CAP, YAW_CAP = 1.2, .15
WHEEL_KD, ACTION_CLIP = 6.0, 6.0
DT = 0.02
LOAD_N = 5.0
TIERS = [0.25, 0.6, 1.0]
DRIVE_S, BRAKE_HOLD_S, BRAKE_S = 10.0, 5.0, 5.0


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
    cfg.episode_length_s = 40.
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
    ap.add_argument("--seed", type=int, default=7393)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=1)
    add_launcher_args(ap)
    args = ap.parse_args()
    for a in ("usd", "urdf", "ckpt"):
        setattr(args, a, getattr(args, a).resolve(strict=True))
    args.out.mkdir(parents=True, exist_ok=True)
    args.physics = "isaacsim_physx"

    with launch_simulation(SimulationCfg(device=args.device, dt=.005), args) as physics:
        (env, Wrapper, PPOCfg, check_v, create_runner, handle_dep) = build(args, physics)
        try:
            wrapped = Wrapper(env, clip_actions=ACTION_CLIP)
            robot = env.scene["robot"]
            cmdterm = env.command_manager.get_term("base_velocity")

            ppo = PPOCfg(); ppo.seed, ppo.device = args.seed, args.device
            ppo.max_iterations = 1
            ppo = handle_dep(ppo, check_v())
            runner = create_runner(wrapped, ppo, log_dir=None)
            runner.load(str(args.ckpt), map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)
            print("SPEED_CKPT", args.ckpt, "iter", runner.current_learning_iteration, flush=True)

            def trial(tier, sign):
                env.seed(args.seed); env.reset()
                cmdterm.vel_command_b[:] = 0.0
                obs = wrapped.get_observations()
                for _ in range(int(2.0 / DT)):
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, _r, _d, _i = wrapped.step(a)

                def step_once():
                    nonlocal obs
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, _r, _d, _i = wrapped.step(a)
                    cf = env.scene["contact_forces"].data.net_forces_w.torch[0][[13, 14, 15, 16]]
                    load = torch.linalg.norm(cf, dim=1)
                    return {"vx": float(robot.data.root_lin_vel_b.torch[0, 0]),
                            "wz": float(robot.data.root_ang_vel_b.torch[0, 2]),
                            "up": float(-robot.data.projected_gravity_b.torch[0][2]),
                            "sup": float((load > LOAD_N).all()),
                            "term": int(bool(_d[0]))}

                cmd = tier * sign
                drive = []
                for _ in range(int(DRIVE_S / DT)):
                    cmdterm.vel_command_b[:] = torch.tensor(
                        [[cmd, 0.0, 0.0]], dtype=torch.float32, device=env.device)
                    drive.append(step_once())
                tail = drive[-150:]
                vx_tail = float(np.mean([r["vx"] for r in tail]))
                track = vx_tail / cmd

                # brake: hold the tier for BRAKE_HOLD_S then command zero
                for _ in range(int(BRAKE_HOLD_S / DT)):
                    cmdterm.vel_command_b[:] = torch.tensor(
                        [[cmd, 0.0, 0.0]], dtype=torch.float32, device=env.device)
                    step_once()
                vx_brake0 = float(robot.data.root_lin_vel_b.torch[0, 0])
                brake = []
                for _ in range(int(BRAKE_S / DT)):
                    cmdterm.vel_command_b[:] = 0.0
                    brake.append(step_once())
                dist = float(sum(abs(r["vx"]) for r in brake) * DT)
                vxs = np.abs([r["vx"] for r in brake])
                below = np.where(vxs < 0.02)[0]
                return {
                    "tier": tier, "dir": "fwd" if sign > 0 else "bwd", "cmd": cmd,
                    "vx_tail": vx_tail, "vx_track": track,
                    "drive_upright_min": float(min(r["up"] for r in drive)),
                    "drive_support_frac": float(np.mean([r["sup"] for r in drive])),
                    "drive_terminations": int(sum(r["term"] for r in drive)),
                    "vx_at_brake": vx_brake0,
                    "brake_distance_m": dist,
                    "brake_time_to_0p02_s": (float(below[0] * DT) if below.size else None),
                    "brake_upright_min": float(min(r["up"] for r in brake)),
                    "brake_terminations": int(sum(r["term"] for r in brake)),
                    "brake_final_vx": float(np.mean([r["vx"] for r in brake[-100:]])),
                }

            rep = {"checkpoint": str(args.ckpt), "seed": args.seed,
                   "note": "measurement only; does NOT imply any tier passed the acceptance gate",
                   "tiers_mps": TIERS, "trials": []}
            for tier in TIERS:
                for sign in (1, -1):
                    t = trial(tier, sign)
                    rep["trials"].append(t)
                    print("  %.2f %s  vx=%+.3f track=%+.2f up=%.3f sup=%.3f term=%d | "
                          "brake: %.3f m, t_stop=%s s, final=%+.4f"
                          % (tier, t["dir"], t["vx_tail"], t["vx_track"],
                             t["drive_upright_min"], t["drive_support_frac"],
                             t["drive_terminations"], t["brake_distance_m"],
                             ("%.2f" % t["brake_time_to_0p02_s"]) if t["brake_time_to_0p02_s"] is not None else "never",
                             t["brake_final_vx"]), flush=True)
            (args.out / "speed_envelope.json").write_text(json.dumps(rep, indent=1, default=str))
            print("SPEED_DONE", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
