#!/usr/bin/env python3
"""Continuous joystick sequence regression: fixed stand -> reverse -> release -> stop -> fixed stand.

The selection standard requires the reverse brake to be tested BOTH standalone and inside a
continuous joystick sequence, and it requires braking to complete three stages on one trajectory:
decelerate, come to rest, and restore the standing posture.  The standalone brake case only covers
the middle of that.  This script runs the whole sequence and judges all three stages plus the usual
hard constraints.

Timeline (one trajectory, continuous, no reset between stages):

    0.0 -  3.0 s   fixed stand, zero command
    3.0 -  4.0 s   ramp the reverse command down to -0.25 m/s
    4.0 -  9.0 s   hold reverse
    9.0 - 19.0 s   release: command to zero, decelerate, come to rest and hold
   19.0 - 22.0 s   restore fixed stand and check the posture error

Judged:
  decelerate  - time and distance from release to |vx| < 0.02
  come to rest- tail |vx| and |wz| over the last 3 s of the hold
  restore     - leg joint error against the default stance at the end, plus upright
  safety      - no termination, no non-wheel contact, four-wheel support, upright
Nothing is tuned; the gates are the frozen ones already in use.
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
STOP_VX, STOP_WZ = 0.02, 0.02
UPRIGHT_MIN = 0.85
SUPPORT_MIN = 0.95
STAND_POS_TOL = 0.20      # rad, max leg joint error at the end of the restore stage
SEQ = [("stand_in", 3.0, 0.0), ("ramp_rev", 1.0, -0.25), ("hold_rev", 5.0, -0.25),
       ("release", 10.0, 0.0), ("stand_out", 3.0, 0.0)]


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
    ap.add_argument("--yaw-kp", type=float, default=0.0)
    ap.add_argument("--ff-cap", type=float, default=4.0)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=1)
    add_launcher_args(ap)
    args = ap.parse_args()
    for k in ("usd", "urdf", "ckpt"):
        setattr(args, k, getattr(args, k).resolve(strict=True))
    args.out.mkdir(parents=True, exist_ok=True)
    args.physics = "isaacsim_physx"

    with launch_simulation(SimulationCfg(device=args.device, dt=.005), args) as physics:
        env, Wrapper, PPOCfg, check_v, create_runner, handle_dep = build(args, physics)
        try:
            wrapped = Wrapper(env, clip_actions=ACTION_CLIP)
            robot = env.scene["robot"]
            cmdterm = env.command_manager.get_term("base_velocity")
            names = list(robot.joint_names)
            bodies = list(robot.body_names)
            wbid = [bodies.index(n) for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")]
            leg_ids = list(range(12))

            ppo = PPOCfg(); ppo.seed, ppo.device = args.seed, args.device
            ppo.max_iterations = 1
            ppo = handle_dep(ppo, check_v())
            runner = create_runner(wrapped, ppo, log_dir=None)
            runner.load(str(args.ckpt), map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)

            env.seed(args.seed); env.reset()
            cmdterm.vel_command_b[:] = 0.0
            obs = wrapped.get_observations()
            rows, term = [], 0
            for stage, secs, vx in SEQ:
                for _ in range(int(secs / DT)):
                    cmdterm.vel_command_b[:] = torch.tensor(
                        [[vx, 0.0, 0.0]], dtype=torch.float32, device=env.device)
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, _r, _d, _i = wrapped.step(a)
                    cf = env.scene["contact_forces"].data.net_forces_w.torch[0][wbid]
                    load = torch.linalg.norm(cf, dim=1)
                    qerr = (robot.data.joint_pos.torch[0][leg_ids]
                            - robot.data.default_joint_pos.torch[0][leg_ids]).abs()
                    rows.append({
                        "stage": stage,
                        "t": len(rows) * DT,
                        "vx": float(robot.data.root_lin_vel_b.torch[0, 0]),
                        "wz": float(robot.data.root_ang_vel_b.torch[0, 2]),
                        "up": float(-robot.data.projected_gravity_b.torch[0][2]),
                        "sup": bool((load > LOAD_N).all()),
                        "leg_err": float(qerr.max()),
                        "term": bool(_d[0]),
                    })
                    term += int(bool(_d[0]))

            def seg(name):
                return [r for r in rows if r["stage"] == name]

            rel = seg("release")
            stop_idx = next((i for i, r in enumerate(rel) if abs(r["vx"]) < STOP_VX), None)
            dist = float(sum(abs(r["vx"]) for r in rel[:stop_idx + 1]) * DT) if stop_idx is not None else None
            tail = rel[-150:] if len(rel) >= 150 else rel
            out_last = seg("stand_out")[-150:]
            res = {
                "checkpoint": str(args.ckpt), "seed": args.seed,
                "yaw_kp": args.yaw_kp, "ff_cap": args.ff_cap,
                "gates": {"stop_vx": STOP_VX, "stop_wz": STOP_WZ, "upright": UPRIGHT_MIN,
                           "support": SUPPORT_MIN, "stand_pos_tol_rad": STAND_POS_TOL},
                "sequence": [{"stage": s, "seconds": d, "cmd_vx": v} for s, d, v in SEQ],
                "release": {
                    "stop_time_s": (stop_idx * DT) if stop_idx is not None else None,
                    "stop_distance_m": dist,
                    "tail_vx": float(np.mean([r["vx"] for r in tail])),
                    "tail_wz": float(np.mean([r["wz"] for r in tail])),
                    "tail_support": float(np.mean([r["sup"] for r in tail])),
                },
                "restore": {
                    "final_leg_err_max": float(max(r["leg_err"] for r in out_last)) if out_last else None,
                    "final_vx": float(np.mean([r["vx"] for r in out_last])) if out_last else None,
                    "final_wz": float(np.mean([r["wz"] for r in out_last])) if out_last else None,
                },
                "safety": {
                    "terminations": term,
                    "upright_min": float(min(r["up"] for r in rows)),
                    "support_min_tail_release": float(np.mean([r["sup"] for r in tail])),
                    "frames": len(rows),
                },
            }
            r, st, sa = res["release"], res["restore"], res["safety"]
            # release-derived values live in r, restore-derived in st; keep them straight
            res["pass"] = bool(
                term == 0
                and sa["upright_min"] >= UPRIGHT_MIN
                and r["tail_support"] >= SUPPORT_MIN
                and abs(r["tail_vx"]) < STOP_VX and abs(r["tail_wz"]) < STOP_WZ
                and r["stop_time_s"] is not None
                and st["final_leg_err_max"] is not None
                and st["final_leg_err_max"] <= STAND_POS_TOL
            )
            print("JOYSTICK_SEQ pass=%s  stop_t=%s s stop_d=%s m  tail_vx=%+.4f tail_wz=%+.4f "
                  "sup=%.3f up_min=%.3f leg_err_final=%.3f term=%d"
                  % (res["pass"], r["stop_time_s"], r["stop_distance_m"],
                     r["tail_vx"], r["tail_wz"], r["tail_support"],
                     sa["upright_min"], st["final_leg_err_max"], term), flush=True)
            (args.out / "joystick_seq.json").write_text(json.dumps(res, indent=1, default=str))
            print("JOYSTICK_SEQ_DONE", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
