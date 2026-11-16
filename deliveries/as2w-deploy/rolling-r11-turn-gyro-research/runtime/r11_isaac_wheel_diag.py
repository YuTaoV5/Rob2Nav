#!/usr/bin/env python3
"""Isaac small-command wheel diagnosis: per-wheel target, actual speed, and ACTUAL solver torque.

The operator's instruction: re-check per-wheel final targets, actual wheel speeds and drive
response for small commands, and settle where the 4.47 Nm figure came from before claiming a
dead-zone.  That figure was reconstructed as KD*(target-actual); here we try to read the force the
solver actually projects onto each wheel DOF, and only fall back to the reconstruction (clearly
labelled) if the API is unavailable.

Reports per wheel, for wz = +-0.05 and +-0.10:
  * mean / max |target|, mean / max |actual dq|, tracking ratio and error
  * reconstructed torque KD*(target-actual) and its max, plus the fraction at the 13.5 Nm cap
  * the ACTUAL projected joint force if readable, with its max, cap fraction, and the ratio
    between actual and reconstructed
Also prints which force getters the live view exposes, so the API question is answered by the run
rather than assumed.
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
WHEEL_TAU_CAP = 13.5
DT = 0.02
TAIL = 150          # last 3 s


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
    cfg.episode_length_s = 40.
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
    ap.add_argument("--settle-s", type=float, default=2.0)
    # robust_config/apply_safety read these off the args namespace, same as the acceptance script
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
            view = robot.root_physx_view

            # --- API discovery: answer the "can we read the real force" question by observing ---
            getters = sorted(m for m in dir(view) if m.startswith("get_") and
                             any(w in m for w in ("force", "torque", "joint")))
            print("VIEW_FORCE_GETTERS %s" % json.dumps(getters), flush=True)
            has_proj = hasattr(view, "get_dof_projected_joint_forces")
            has_incoming = hasattr(view, "get_link_incoming_joint_force")
            print("VIEW_FORCE_API projected=%s incoming=%s" % (has_proj, has_incoming), flush=True)

            ppo = PPOCfg(); ppo.seed, ppo.device = args.seed, args.device
            ppo.max_iterations = 1
            ppo = handle_dep(ppo, check_v())
            runner = create_runner(wrapped, ppo, log_dir=None)
            runner.load(str(args.ckpt), map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)
            print("DIAG_CKPT %s iter %d" % (args.ckpt, runner.current_learning_iteration), flush=True)

            # wheel joint indices by NAME, not by position
            names = list(robot.joint_names)
            wheel_idx = [names.index(n) for n in
                         ("FL_foot", "FR_foot", "RL_foot", "RR_foot")]
            print("WHEEL_JOINT_NAMES %s" % json.dumps([names[i] for i in wheel_idx]), flush=True)

            def read_actual():
                """Return the solver-projected wheel torques, or None if the API is absent."""
                if not has_proj:
                    return None, "api_absent"
                try:
                    f = view.get_dof_projected_joint_forces()
                    t = f.torch if hasattr(f, "torch") else torch.as_tensor(f, device=env.device)
                    return t[0][wheel_idx].detach().cpu().numpy().astype(float), "measured"
                except Exception as exc:
                    return None, "%s: %s" % (type(exc).__name__, exc)

            def trial(wz, seconds=12.0):
                env.seed(args.seed); env.reset()
                cmdterm.vel_command_b[:] = 0.0
                obs = wrapped.get_observations()
                for _ in range(int(args.settle_s / DT)):
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, _r, _d, _i = wrapped.step(a)
                rows = []
                for _ in range(int(seconds / DT)):
                    cmdterm.vel_command_b[:] = torch.tensor(
                        [[0.0, 0.0, wz]], dtype=torch.float32, device=env.device)
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, _r, _d, _i = wrapped.step(a)
                    # the action manager holds the post-clip, post-decimation target actually sent
                    act = env.action_manager.action[0][wheel_idx].detach().cpu().numpy().copy()
                    dq = robot.data.joint_vel.torch[0][wheel_idx].detach().cpu().numpy().copy()
                    tgt = act * 16.0          # wheel scale
                    recon = WHEEL_KD * (tgt - dq)
                    act_f, act_src = read_actual()
                    rows.append({"tgt": tgt, "dq": dq, "recon": recon,
                                 "actual": act_f, "src": act_src,
                                 "wz": float(robot.data.root_ang_vel_b.torch[0, 2])})
                return rows, act_src

            def summarise(rows, src):
                tgt = np.array([r["tgt"] for r in rows])
                dq = np.array([r["dq"] for r in rows])
                rec = np.array([r["recon"] for r in rows])
                yaw = np.array([r["wz"] for r in rows])
                n = len(rows); tl = slice(n - TAIL, n)
                out = {"n_frames": n, "actual_source": src,
                       "yaw_track_tail": None,
                       "per_wheel": {}}
                for i, w in enumerate(("FL", "FR", "RL", "RR")):
                    e = {
                        "tgt_mean": float(tgt[tl, i].mean()),
                        "tgt_absmax": float(np.abs(tgt[:, i]).max()),
                        "dq_mean": float(dq[tl, i].mean()),
                        "recon_mean": float(rec[tl, i].mean()),
                        "recon_absmax": float(np.abs(rec[:, i]).max()),
                        "recon_cap_frac": float((np.abs(rec[:, i]) >= WHEEL_TAU_CAP * 0.999).mean()),
                        "track_ratio": (float(dq[tl, i].mean() / tgt[tl, i].mean())
                                        if abs(tgt[tl, i].mean()) > 1e-9 else None),
                    }
                    if rows[0]["actual"] is not None:
                        fa = np.array([r["actual"] for r in rows])
                        e.update({
                            "actual_mean": float(fa[tl, i].mean()),
                            "actual_absmax": float(np.abs(fa[:, i]).max()),
                            "actual_cap_frac": float((np.abs(fa[:, i]) >= WHEEL_TAU_CAP * 0.999).mean()),
                            "actual_over_recon": (float(np.abs(fa[tl, i]).mean() / np.abs(rec[tl, i]).mean())
                                                  if np.abs(rec[tl, i]).mean() > 1e-9 else None),
                        })
                    out["per_wheel"][w] = e
                if abs(wz) > 1e-9:
                    out["yaw_track_tail"] = float(yaw[tl].mean() / wz)
                return out

            rep = {"checkpoint": str(args.ckpt), "seed": args.seed,
                   "wheel_tau_cap_nm": WHEEL_TAU_CAP, "wheel_kd": WHEEL_KD, "wheel_scale": 16.0,
                   "api": {"getters": getters, "projected_available": has_proj,
                           "incoming_available": has_incoming},
                   "note": "reconstruction is KD*(target-actual); 'actual' is the solver-projected DOF force when the API allows",
                   "cases": {}}
            for wz in (0.05, -0.05, 0.10, -0.10):
                name = "turn_%s_%.2f" % ("left" if wz > 0 else "right", abs(wz))
                rows, src = trial(wz)
                s = summarise(rows, src)
                s["cmd_wz"] = wz
                rep["cases"][name] = s
                pw = s["per_wheel"]
                print("== %-16s src=%-10s yaw_track_tail=%+.2f"
                      % (name, src, s["yaw_track_tail"] or 0), flush=True)
                for w in ("FL", "FR", "RL", "RR"):
                    e = pw[w]
                    extra = (" actual_max=%+.2f cap_fr=%.3f a/r=%.2f"
                             % (e["actual_absmax"], e["actual_cap_frac"], e["actual_over_recon"] or 0)
                             ) if "actual_absmax" in e else ""
                    print("    %s tgt=%+7.2f dq=%+7.2f recon=%+7.2f recon_max=%+8.2f cap_fr=%.3f%s"
                          % (w, e["tgt_mean"], e["dq_mean"], e["recon_mean"],
                             e["recon_absmax"], e["recon_cap_frac"], extra), flush=True)
            (args.out / "isaac_wheel_diag.json").write_text(json.dumps(rep, indent=1, default=str))
            print("ISAAC_WHEEL_DIAG_DONE", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
