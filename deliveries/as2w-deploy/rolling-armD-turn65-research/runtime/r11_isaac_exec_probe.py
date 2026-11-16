#!/usr/bin/env python3
"""AS2-W rolling: probe the ISAAC wheel execution chain per wheel, per frame.

Why
  GOAL_STATUS 69: the same checkpoint (armE r11-final) scores 7/11 on MuJoCo but
  5/11 on Isaac.  Arc yaw is 1.04 / 0.84 on MuJoCo versus 0.66 / 0.57 on Isaac, and
  the pure turns produce essentially no yaw on Isaac (0.01-0.22) even though Isaac's
  four-wheel support is a clean 1.000.  Earlier measurements put Isaac's differential
  delivery at exec 0.43-0.57 versus MuJoCo's 0.86-1.04.

  Before changing any gain, integral or feedforward, the MECHANISM has to be known:
    (a) is the wheel drive TORQUE-SATURATED (hitting the +-13.5 Nm cap), or
    (b) is it torque-limited but far below the cap, i.e. simply too little gain for
        the steady-state load (a steady-state velocity error)?
  Those two need opposite fixes, so this probe measures, per wheel and per frame:
      target velocity (what the policy asks), actual velocity, the kd-law torque
      6*(target-actual), whether it saturates, and the resulting differential.

Output
  JSON with per-case, per-wheel: mean |target|, mean |actual|, mean signed error,
  mean |torque|, max |torque|, fraction of frames at the torque cap, and the
  differential (d4) of each.  No gates, no pass/fail - this is a measurement.

Contract
  Same 65-dim obs / 16-dim action / 20 ms policy step / wheel kp=0 kd=6
  +-13.5 Nm as r11_acceptance_v2.py; the wheel target is taken from the action
  manager's delayed target exactly as the deployment wrapper does.
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
WHEEL_KD, WHEEL_SCALE, ACTION_CLIP = 6.0, 16.0, 6.0
WHEEL_LIMIT = 13.5            # 0.9 x 15 Nm hard cap
DT = 0.02
SETTLE_S = 2.0
TAIL_S = 3.0


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


def d4(A):
    A = np.asarray(A, float)
    return A[:, 1] + A[:, 3] - A[:, 0] - A[:, 2]


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
            wj = env.as2w_joint_ids[12:]
            cmdterm = env.command_manager.get_term("base_velocity")
            wterm = env.action_manager.get_term("wheel_vel")

            ppo = PPOCfg(); ppo.seed, ppo.device = args.seed, args.device
            ppo.max_iterations = 1
            ppo = handle_dep(ppo, check_v())
            runner = create_runner(wrapped, ppo, log_dir=None)
            runner.load(str(args.ckpt), map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)
            print("PROBE_CKPT", args.ckpt, "iter", runner.current_learning_iteration, flush=True)

            def settle():
                env.seed(args.seed); env.reset()
                cmdterm.vel_command_b[:] = 0.0
                obs = wrapped.get_observations()
                for _ in range(int(SETTLE_S / DT)):
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, *_ = wrapped.step(a)
                return obs

            def run(cmd_fn, seconds):
                obs = settle()
                n = int(seconds / DT)
                T, V, TAU, UP = [], [], [], []
                LOAD = []
                for k in range(n):
                    c = cmd_fn(k * DT)
                    cmdterm.vel_command_b[:] = torch.tensor(
                        [c], dtype=torch.float32, device=env.device)
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, *_ = wrapped.step(a)
                    tgt = wterm.delayed()[0].detach()
                    act = robot.data.joint_vel.torch[0, wj].detach()
                    tau = WHEEL_KD * (tgt - act)
                    T.append(tgt.cpu().numpy()); V.append(act.cpu().numpy())
                    TAU.append(tau.cpu().numpy())
                    g = robot.data.projected_gravity_b.torch[0]
                    UP.append(float(-g[2]))
                    cf = env.scene["contact_forces"].data.net_forces_w.torch[0][[13, 14, 15, 16]]
                    LOAD.append(torch.linalg.norm(cf, dim=1).detach().cpu().numpy())
                return (np.asarray(T), np.asarray(V), np.asarray(TAU), np.asarray(UP),
                        np.asarray(LOAD))

            def stats(C, cmd):
                tail = max(1, int(TAIL_S / DT))
                T, V, TAU, UP, LOAD = C
                tt, vv, aa = T[-tail:], V[-tail:], TAU[-tail:]
                sat = (np.abs(aa) >= WHEEL_LIMIT * 0.98).mean()
                return {
                    "cmd": [float(x) for x in cmd],
                    "per_wheel": {
                        w: {
                            "target_abs_mean": float(np.abs(tt[:, i]).mean()),
                            "actual_abs_mean": float(np.abs(vv[:, i]).mean()),
                            "signed_err_mean": float((tt[:, i] - vv[:, i]).mean()),
                            "torque_abs_mean": float(np.abs(aa[:, i]).mean()),
                            "torque_abs_max": float(np.abs(aa[:, i]).max()),
                            "torque_cap_frac": float((np.abs(aa[:, i]) >= WHEEL_LIMIT * 0.98).mean()),
                        } for i, w in enumerate(("FL", "FR", "RL", "RR"))
                    },
                    "ddq_target_mean": float(d4(tt).mean()),
                    "ddq_actual_mean": float(d4(vv).mean()),
                    "ddq_exec": (float(np.abs(d4(vv)).mean() / np.abs(d4(tt)).mean())
                                 if np.abs(d4(tt)).mean() > 1e-9 else None),
                    "torque_cap_frac_any": float(sat),
                    "upright_min": float(UP.min()),
                    "load_n_per_wheel": {
                        w: {"mean": float(LOAD[-tail:, i].mean()),
                            "min": float(LOAD[-tail:, i].min()),
                            "max": float(LOAD[-tail:, i].max())}
                        for i, w in enumerate(("FL", "FR", "RL", "RR"))},
                    "load_total_mean_n": float(LOAD[-tail:].sum(axis=1).mean()),
                    "robot_mass_kg": float(np.asarray(
                        robot.root_physx_view.get_masses()).sum()),
                }

            def const(vx, wz):
                return lambda t: [vx, 0.0, wz]

            CASES = [("turn_right_0.05", 12.0, const(0.0, -0.05)),
                     ("turn_left_0.05", 12.0, const(0.0, +0.05)),
                     ("arc_left_0.25", 12.0, const(0.25, +0.05)),
                     ("arc_right_0.25", 12.0, const(0.25, -0.05))]

            report = {"checkpoint": str(args.ckpt), "seed": args.seed,
                      "wheel_kd": WHEEL_KD, "wheel_limit_nm": WHEEL_LIMIT,
                      "contract": "same as r11_acceptance_v2", "cases": {}}
            for name, secs, fn in CASES:
                C = run(fn, secs)
                s = stats(C, fn(0.0))
                report["cases"][name] = s
                print()
                print("== %s ==" % name, flush=True)
                print("  ddq target=%+.4f actual=%+.4f exec=%s  torque_cap_frac=%.3f"
                      % (s["ddq_target_mean"], s["ddq_actual_mean"],
                         ("%.3f" % s["ddq_exec"]) if s["ddq_exec"] else "-",
                         s["torque_cap_frac_any"]), flush=True)
                print("  load_total=%.1f N  mass=%.2f kg  per-wheel mean N: %s"
                      % (s["load_total_mean_n"], s["robot_mass_kg"],
                         " ".join("%s=%.1f" % (w, s["load_n_per_wheel"][w]["mean"])
                                  for w in ("FL", "FR", "RL", "RR"))), flush=True)
                for w in ("FL", "FR", "RL", "RR"):
                    d = s["per_wheel"][w]
                    print("    %s target=%+.3f actual=%+.3f err=%+.3f tau=%+.2f Nm (max %.2f, cap %.1f%%)"
                          % (w, d["target_abs_mean"] * np.sign(d["signed_err_mean"] or 1),
                             d["actual_abs_mean"] * np.sign(d["signed_err_mean"] or 1),
                             d["signed_err_mean"], d["torque_abs_mean"],
                             d["torque_abs_max"], 100 * d["torque_cap_frac"]), flush=True)
            (args.out / "exec_probe.json").write_text(json.dumps(report, indent=1, default=str))
            print()
            print("EXEC_PROBE_DONE", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
