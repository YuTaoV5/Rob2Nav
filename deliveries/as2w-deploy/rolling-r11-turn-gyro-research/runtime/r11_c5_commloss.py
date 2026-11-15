#!/usr/bin/env python3
"""C5 safety addendum: what happens when the command link is lost.

Checklist v3 section 4 requires a verified safe reaction to communication loss
("通讯失效 ... 执行已验证的安全处置") and section 5 requires the loss/gain of the link to be
covered.  This probe compares the two plausible loss-handling policies on the same
checkpoint, in simulation only:

  hold_last  the command is frozen at its last received value (a naive driver)
  zero       the command is forced to zero as soon as the link is declared lost

Sequence per variant:
  drive   6 s at vx 0.25           (steady state)
  loss    3 s with the variant's handling
  recover 5 s at zero command      (the verified stop the goal actually wants)

Reported per variant: speed at the moment of loss, distance travelled while the link was
down, time to fall below 0.02 m/s after the recovery command, final drift, terminations and
support.  No pass/fail gate: the acceptance matrix owns verdicts; this is the safety
evidence and it is simulation-only.
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
LOAD_N = 5.0
DRIVE_S, LOSS_S, RECOVER_S = 6.0, 3.0, 5.0
STOP_EPS = 0.02


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
    ap.add_argument("--vx", type=float, default=0.25)
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

            ppo = PPOCfg(); ppo.seed, ppo.device = args.seed, args.device
            ppo.max_iterations = 1
            ppo = handle_dep(ppo, check_v())
            runner = create_runner(wrapped, ppo, log_dir=None)
            runner.load(str(args.ckpt), map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)
            print("C5B_CKPT", args.ckpt, "iter", runner.current_learning_iteration, flush=True)

            def variant(mode):
                env.seed(args.seed); env.reset()
                cmdterm.vel_command_b[:] = 0.0
                obs = wrapped.get_observations()
                for _ in range(int(2.0 / DT)):
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, _r, _d, _i = wrapped.step(a)

                def cmd(vx):
                    cmdterm.vel_command_b[:] = torch.tensor(
                        [[vx, 0.0, 0.0]], dtype=torch.float32, device=env.device)

                def step_once():
                    nonlocal obs
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, _r, _d, _i = wrapped.step(a)
                    cf = env.scene["contact_forces"].data.net_forces_w.torch[0][[13, 14, 15, 16]]
                    load = torch.linalg.norm(cf, dim=1)
                    return {
                        "vx": float(robot.data.root_lin_vel_b.torch[0, 0]),
                        "wz": float(robot.data.root_ang_vel_b.torch[0, 2]),
                        "upright": float(-robot.data.projected_gravity_b.torch[0][2]),
                        "support": float((load > LOAD_N).all()),
                        "term": int(bool(_d[0])),
                    }

                n = int(DRIVE_S / DT)
                for _ in range(n):
                    cmd(args.vx)          # must be re-asserted every step: the manager lerps
                    step_once()
                vx_at_loss = float(robot.data.root_lin_vel_b.torch[0, 0])

                # link lost: the two candidate policies differ only here
                if mode == "zero":
                    cmd(0.0)
                # hold_last leaves the command at its last value

                lost = []
                for _ in range(int(LOSS_S / DT)):
                    if mode == "hold_last":
                        cmd(args.vx)
                    lost.append(step_once())

                rec = []
                for _ in range(int(RECOVER_S / DT)):
                    cmd(0.0)
                    rec.append(step_once())

                vxs = np.abs([r["vx"] for r in rec])
                below = np.where(vxs < STOP_EPS)[0]
                t_stop = (float(below[0]) * DT) if below.size else None
                return {
                    "mode": mode,
                    "vx_at_loss": vx_at_loss,
                    "loss_speed_mean": float(np.mean([abs(r["vx"]) for r in lost])),
                    "loss_distance_m": float(sum(abs(r["vx"]) for r in lost) * DT),
                    "loss_upright_min": float(min(r["upright"] for r in lost)),
                    "loss_support_frac": float(np.mean([r["support"] for r in lost])),
                    "loss_terminations": int(sum(r["term"] for r in lost)),
                    "time_to_below_0p02_s": t_stop,
                    "recover_vx_tail": float(np.mean([r["vx"] for r in rec[-150:]])),
                    "recover_wz_tail": float(np.mean([r["wz"] for r in rec[-150:]])),
                    "recover_terminations": int(sum(r["term"] for r in rec)),
                }

            rep = {"checkpoint": str(args.ckpt), "seed": args.seed,
                   "command_vx": args.vx, "note": "simulation only; not hardware evidence",
                   "sequence": {"drive_s": DRIVE_S, "loss_s": LOSS_S, "recover_s": RECOVER_S},
                   "variants": [variant("hold_last"), variant("zero")]}
            (args.out / "c5_commloss.json").write_text(json.dumps(rep, indent=1, default=str))
            print()
            for v in rep["variants"]:
                print("  %-9s vx@loss=%+.3f loss_speed=%.3f loss_dist=%.3f m "
                      "loss_up=%.3f loss_sup=%.3f term=%d | t_stop=%s s | "
                      "recover vx=%+.4f wz=%+.4f term=%d"
                      % (v["mode"], v["vx_at_loss"], v["loss_speed_mean"], v["loss_distance_m"],
                         v["loss_upright_min"], v["loss_support_frac"], v["loss_terminations"],
                         ("%.2f" % v["time_to_below_0p02_s"]) if v["time_to_below_0p02_s"] is not None else "never",
                         v["recover_vx_tail"], v["recover_wz_tail"], v["recover_terminations"]),
                      flush=True)
            print("C5B_DONE", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
