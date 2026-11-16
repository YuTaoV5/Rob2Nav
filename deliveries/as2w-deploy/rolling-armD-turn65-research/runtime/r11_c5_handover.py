#!/usr/bin/env python3
"""C5 hand-over probe: fixed stand -> restricted ramp -> command -> ramp down -> stand.

Checklist v3 C5 asks for the allowed poses and speeds, the history initialisation, the success
and failure conditions, the exit target, and the control-ownership hand-over.  This probe
exercises the parts that can be exercised in simulation and reports the numbers a hand-over
review needs, without claiming anything about hardware:

  phase 1  stand      5 s at zero command  (the rolling stand is the entry and exit pose)
  phase 2  ramp up    2 s zero -> command  (restricted transition; the command manager itself
                                            also lerps at 0.12 per step, so this is a
                                            command ramp on top of that)
  phase 3  hold cmd   8 s at the command
  phase 4  ramp down  2 s command -> zero
  phase 5  stand      5 s at zero command  (the exit pose)

Reported per phase and for the transitions:
  * upright min, four-wheel support fraction, terminations
  * max |dq| per wheel and per leg joint at the takeover instant (continuity)
  * drift over the final stand window (|vx|, |wz|, accumulated position and yaw)
  * angle-error peaks against the default pose
No gates and no pass/fail: the acceptance matrix owns the verdicts; this is the hand-over
evidence C5 asks for, and it is explicitly simulation-only.
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
STAGES = [("stand_in", 5.0), ("ramp_up", 2.0), ("hold", 8.0),
          ("ramp_down", 2.0), ("stand_out", 5.0)]


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
    ap.add_argument("--wz", type=float, default=0.0)
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
            lj = env.as2w_joint_ids[:12]
            cmdterm = env.command_manager.get_term("base_velocity")
            qdef = robot.data.default_joint_pos.torch[0, env.as2w_joint_ids].clone()

            ppo = PPOCfg(); ppo.seed, ppo.device = args.seed, args.device
            ppo.max_iterations = 1
            ppo = handle_dep(ppo, check_v())
            runner = create_runner(wrapped, ppo, log_dir=None)
            runner.load(str(args.ckpt), map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)
            print("C5_CKPT", args.ckpt, "iter", runner.current_learning_iteration, flush=True)

            env.seed(args.seed); env.reset()
            # warm the policy's own history with a short zero-command stand
            cmdterm.vel_command_b[:] = 0.0
            obs = wrapped.get_observations()
            for _ in range(int(2.0 / DT)):
                with torch.inference_mode():
                    a = policy(obs)
                obs, _r, _d, _i = wrapped.step(a)

            pos0 = robot.data.root_pos_w.torch[0].clone()
            yaw0 = float(robot.data.heading_w[0]) if hasattr(robot.data, "heading_w") else 0.0

            rows = []
            takeover = {}
            for si, (name, secs) in enumerate(STAGES):
                n = int(secs / DT)
                for k in range(n):
                    frac = k / max(1, n - 1)
                    if name == "stand_in" or name == "stand_out":
                        vx = wz = 0.0
                    elif name == "hold":
                        vx, wz = args.vx, args.wz
                    elif name == "ramp_up":
                        vx, wz = args.vx * frac, args.wz * frac
                    else:
                        vx, wz = args.vx * (1.0 - frac), args.wz * (1.0 - frac)
                    cmdterm.vel_command_b[:] = torch.tensor(
                        [[vx, 0.0, wz]], dtype=torch.float32, device=env.device)
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, _r, _d, _i = wrapped.step(a)
                    if name == "hold" and k == 0:
                        takeover = {
                            "first_frame_wheel_dq": [float(v) for v in
                                                     robot.data.joint_vel.torch[0, wj]],
                            "first_frame_leg_dq": [float(v) for v in
                                                   robot.data.joint_vel.torch[0, lj]],
                            "first_frame_leg_q_err": [float(v) for v in
                                                      (robot.data.joint_pos.torch[0, lj] - qdef[:12])],
                        }
                    cf = env.scene["contact_forces"].data.net_forces_w.torch[0][[13, 14, 15, 16]]
                    load = torch.linalg.norm(cf, dim=1)
                    rows.append({
                        "stage": name, "k": k, "t_s": round((si * 0 + k) * DT, 3),
                        "vx_cmd": vx, "wz_cmd": wz,
                        "vx": float(robot.data.root_lin_vel_b.torch[0, 0]),
                        "wz": float(robot.data.root_ang_vel_b.torch[0, 2]),
                        "yaw": float(-robot.data.projected_gravity_b.torch[0][2]),
                        "support": float((load > LOAD_N).all()),
                        "term": int(bool(_d[0])),
                        "leg_q_err_absmax": float((robot.data.joint_pos.torch[0, lj]
                                                   - qdef[:12]).abs().max()),
                        "leg_dq_absmax": float(robot.data.joint_vel.torch[0, lj].abs().max()),
                        "wheel_dq_absmax": float(robot.data.joint_vel.torch[0, wj].abs().max()),
                    })

            by = {}
            for name, secs in STAGES:
                seg = [r for r in rows if r["stage"] == name]
                if not seg:
                    continue
                tail = max(1, int(3.0 / DT))
                by[name] = {
                    "seconds": secs,
                    "upright_min": float(min(r["yaw"] for r in seg)),
                    "support_frac": float(np.mean([r["support"] for r in seg])),
                    "terminations": int(sum(r["term"] for r in seg)),
                    "vx_tail_mean": float(np.mean([r["vx"] for r in seg[-tail:]])),
                    "wz_tail_mean": float(np.mean([r["wz"] for r in seg[-tail:]])),
                    "leg_q_err_absmax": float(max(r["leg_q_err_absmax"] for r in seg)),
                    "leg_dq_absmax": float(max(r["leg_dq_absmax"] for r in seg)),
                    "wheel_dq_absmax": float(max(r["wheel_dq_absmax"] for r in seg)),
                }
            rep = {
                "checkpoint": str(args.ckpt), "seed": args.seed,
                "command": {"vx": args.vx, "wz": args.wz},
                "sequence": [{"stage": n, "seconds": s} for n, s in STAGES],
                "note": "simulation only; not hardware evidence",
                "phases": by,
                "takeover_first_frame": takeover,
                "final_stand": {
                    "drift_vx_abs": abs(by.get("stand_out", {}).get("vx_tail_mean", 0.0)),
                    "drift_wz_abs": abs(by.get("stand_out", {}).get("wz_tail_mean", 0.0)),
                    "displacement_m": float((robot.data.root_pos_w.torch[0] - pos0).norm()),
                },
            }
            (args.out / "c5_handover.json").write_text(json.dumps(rep, indent=1, default=str))
            print()
            for name, _s in STAGES:
                if name not in by:
                    continue
                d = by[name]
                print("  %-10s up=%.3f sup=%.3f term=%d |vx_tail|=%+.4f |wz_tail|=%+.4f "
                      "legq=%.3f legdq=%.2f wheeldq=%.2f"
                      % (name, d["upright_min"], d["support_frac"], d["terminations"],
                         d["vx_tail_mean"], d["wz_tail_mean"], d["leg_q_err_absmax"],
                         d["leg_dq_absmax"], d["wheel_dq_absmax"]), flush=True)
            fs = rep["final_stand"]
            print("  final drift: |vx|=%.4f  |wz|=%.4f  displacement=%.3f m"
                  % (fs["drift_vx_abs"], fs["drift_wz_abs"], fs["displacement_m"]), flush=True)
            print("C5_DONE", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
