#!/usr/bin/env python3
"""Isaac four-metric probe: joint error, wheel-centre error, zero-command drift, restart support.

The operator asked for the posture bias to be compared "as-is" in Isaac Lab, judged on four things
together.  The acceptance matrix cannot do that: it measures support and yaw but not the wheel-centre
placement, not a zero-command drift, and not a restart takeover.  This probe runs the whole sequence

    settle -> turn -> release -> return-to-stand -> restart

and reports, per leg, the joint error and the wheel-centre delta in the BODY frame (the quantity the
operator cares about because a joint error is not a footfall error), plus the zero-command drift of the
body and the support fraction of the restart phase.

Run twice from the same initial state, once with --leg-bias-rad empty and once with a calibrated vector,
to obtain the as-is comparison.  The bias is applied to the leg part of the action before the wrapper
scales it, exactly as the acceptance script's experimental switch does.
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
LEG_SCALE = 0.35
LOAD_N = 5.0
LEGS = ("FL", "FR", "RL", "RR")
CASES = [("arc_left", 0.25, 0.05), ("arc_right", 0.25, -0.05),
         ("spin_left", 0.0, 0.05), ("spin_right", 0.0, -0.05)]


def build(args, physics):
    from as2w_isaac_flat_env_all import (
        RslRlVecEnvWrapper, UnitreeGo2FlatPPORunnerCfg,
        check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg)
    from as2w_isaac_robust_env import RobustEnvironment, TakeoverCommands, robust_config
    from as2w_training_safety import apply_safety

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
    cfg.episode_length_s = 60.
    cfg.scene.robot.actuators["foot"].damping = WHEEL_KD
    apply_safety(cfg, args.urdf, perturb=False, surface_guard=True)
    env = RobustEnvironment(cfg)
    return env, RslRlVecEnvWrapper, UnitreeGo2FlatPPORunnerCfg, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7393)
    ap.add_argument("--leg-bias-rad", default="")
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--turn-s", type=float, default=12.0)
    ap.add_argument("--release-s", type=float, default=6.0)
    ap.add_argument("--stand-s", type=float, default=4.0)
    ap.add_argument("--restart-s", type=float, default=8.0)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=1)
    add_launcher_args(ap)
    args = ap.parse_args()
    for k in ("usd", "urdf", "ckpt"):
        setattr(args, k, getattr(args, k).resolve(strict=True))
    args.out.mkdir(parents=True, exist_ok=True)
    args.physics = "isaacsim_physx"

    bias = None
    if str(args.leg_bias_rad).strip():
        v = [float(x) for x in str(args.leg_bias_rad).split(",") if x.strip()]
        assert len(v) == 12, "leg_bias_rad needs 12 values"
        bias = torch.tensor(v, dtype=torch.float32) / LEG_SCALE

    with launch_simulation(SimulationCfg(device=args.device, dt=.005), args) as physics:
        env, Wrapper, PPOCfg, check_v, create_runner, handle_dep = build(args, physics)
        try:
            wrapped = Wrapper(env, clip_actions=ACTION_CLIP)
            robot = env.scene["robot"]
            cmdterm = env.command_manager.get_term("base_velocity")
            bodies = list(robot.body_names)
            wb = [bodies.index(f"{L}_foot") for L in LEGS]
            leg_ids = list(range(12))
            qdef = robot.data.default_joint_pos.torch[0][leg_ids].detach().cpu().numpy().copy()

            ppo = PPOCfg(); ppo.seed, ppo.device = args.seed, args.device
            ppo.max_iterations = 1
            ppo = handle_dep(ppo, check_v())
            runner = create_runner(wrapped, ppo, log_dir=None)
            runner.load(str(args.ckpt), map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)

            def step(a):
                if bias is not None:
                    a = a.clone()
                    a[:, :12] = a[:, :12] + bias.to(a.device)
                return wrapped.step(a)

            def wheels_body():
                R = robot.data.root_quat_w.torch[0]
                import torch as _t
                w, x, y, z = [float(v) for v in R]
                M = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                              [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                              [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
                root = robot.data.root_pos_w.torch[0].detach().cpu().numpy()
                pos = robot.data.body_pos_w.torch[0].detach().cpu().numpy()
                return np.array([M.T @ (pos[b] - root) for b in wb])

            def loads():
                cf = env.scene["contact_forces"].data.net_forces_w.torch[0]
                return torch.linalg.norm(cf[wb], dim=1).detach().cpu().numpy()

            rep = {"checkpoint": str(args.ckpt), "seed": args.seed,
                   "leg_bias_rad": args.leg_bias_rad, "bias_active": bias is not None,
                   "load_threshold_n": LOAD_N, "cases": {}}

            for name, vx, wz in CASES:
                env.seed(args.seed); env.reset()
                cmdterm.vel_command_b[:] = 0.0
                obs = wrapped.get_observations()
                for _ in range(int(args.settle_s / DT)):
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, _r, _d, _i = step(a)
                ref_pos = wheels_body()
                q_after_settle_ref = robot.data.joint_pos.torch[0][leg_ids].detach().cpu().numpy().copy()
                body0 = robot.data.root_pos_w.torch[0][:2].detach().cpu().numpy().copy()

                snaps = {}
                def snap(tag):
                    snaps[tag] = {"q": robot.data.joint_pos.torch[0][leg_ids].detach().cpu().numpy().copy(),
                                   "wp": wheels_body(), "load": loads()}

                snap("before_turn")
                for phase_name, secs, cvx, cwz in (("turn", args.turn_s, vx, wz),
                                                    ("release", args.release_s, 0.0, 0.0)):
                    cmdterm.vel_command_b[:] = torch.tensor([[cvx, 0.0, cwz]], dtype=torch.float32, device=env.device)
                    n = int(secs / DT)
                    sup = []
                    drift = []
                    for _ in range(n):
                        with torch.inference_mode():
                            a = policy(obs)
                        obs, _r, _d, _i = step(a)
                        lo = loads()
                        sup.append(bool((lo > LOAD_N).all()))
                        drift.append(float(np.hypot(*(robot.data.root_pos_w.torch[0][:2].detach().cpu().numpy() - body0))))
                    snaps[phase_name + "_stats"] = {"support_frac": float(np.mean(sup)),
                                                     "drift_max_m": float(np.max(drift))}
                    snap(phase_name + "_end")
                # zero-command return-to-stand; drift measured ONLY here
                cmdterm.vel_command_b[:] = 0.0
                drift, sup = [], []
                x0 = robot.data.root_pos_w.torch[0][:2].detach().cpu().numpy().copy()
                for _ in range(int(args.stand_s / DT)):
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, _r, _d, _i = step(a)
                    sup.append(bool((loads() > LOAD_N).all()))
                    drift.append(float(np.hypot(*(robot.data.root_pos_w.torch[0][:2].detach().cpu().numpy() - x0))))
                snap("after_stand")
                # restart
                cmdterm.vel_command_b[:] = torch.tensor([[vx, 0.0, wz]], dtype=torch.float32, device=env.device)
                sup, yaw = [], []
                for _ in range(int(args.restart_s / DT)):
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, _r, _d, _i = step(a)
                    sup.append(bool((loads() > LOAD_N).all()))
                    yaw.append(float(robot.data.root_ang_vel_b.torch[0, 2]))
                snap("after_restart")

                per_leg = {}
                for i, leg in enumerate(LEGS):
                    per_leg[leg] = {}
                    for tag in ("before_turn", "turn_end", "release_end", "after_stand", "after_restart"):
                        if tag not in snaps:
                            continue
                        qe = float(np.abs(snaps[tag]["q"][i * 3:(i + 1) * 3] - qdef[i * 3:(i + 1) * 3]).max())
                        dp = snaps[tag]["wp"][i] - ref_pos[i]
                        per_leg[leg][tag] = {"q_err_max": round(qe, 4),
                                              "wheel_dpos_fwd_m": round(float(dp[0]), 4),
                                              "wheel_dpos_lat_m": round(float(dp[1]), 4),
                                              "load_N": round(float(snaps[tag]["load"][i]), 1)}
                rep["cases"][name] = {
                    "command": {"vx": vx, "wz": wz},
                    "per_leg": per_leg,
                    "zero_command_drift_m": round(float(np.max(drift)), 4),
                    "return_to_stand_support": float(np.mean(sup)),
                    "restart_support_frac": float(np.mean(sup)),
                    "restart_yaw_tail": float(np.mean(yaw[-150:])) if yaw else None,
                    "turn_support_frac": snaps.get("turn_stats", {}).get("support_frac"),
                    "release_support_frac": snaps.get("release_stats", {}).get("support_frac"),
                }
                c = rep["cases"][name]
                print("== %-11s zero_command_drift=%.4f m  restart_support=%.3f  restart_yaw_tail=%s"
                      % (name, c["zero_command_drift_m"], c["restart_support_frac"],
                         None if c["restart_yaw_tail"] is None else round(c["restart_yaw_tail"], 3)), flush=True)
                for leg in LEGS:
                    p = per_leg[leg]
                    print("    %-2s q_err stand=%.3f restart=%.3f | wheel_fwd stand=%+.4f restart=%+.4f"
                          % (leg, p["after_stand"]["q_err_max"], p["after_restart"]["q_err_max"],
                             p["after_stand"]["wheel_dpos_fwd_m"], p["after_restart"]["wheel_dpos_fwd_m"]), flush=True)
            tag = "bias_on" if bias is not None else "bias_off"
            (args.out / f"four_metric_{tag}.json").write_text(json.dumps(rep, indent=1))
            print("FOUR_METRIC_DONE", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
