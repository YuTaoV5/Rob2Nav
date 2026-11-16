#!/usr/bin/env python3
"""Isaac: wheel break-away threshold, redesigned after the one-wheel probe failed.

Why the first probe was invalid: ramping ONE wheel moved the whole robot and dragged the other
three along, so the pre-ramp baseline was already 0.26-0.48 rad/s and "first sample above a
threshold" could not be read as static-friction break-away.

Redesign, as recorded in evidence/WHEEL_BREAKAWAY_PROBE.md:
  1. COMMON-MODE ramp: all four wheel targets rise together, so the robot tracks straight and no
     wheel is dragged by a yaw moment.  Differences between the four wheels at the same commanded
     target then reflect per-wheel load / friction asymmetry, so left-versus-right is obtained from
     one run instead of two.
  2. Baseline gate: the wheels must be below --baseline-dq for the whole settle window, otherwise
     the trial is VOID rather than silently reported.
  3. Fine steps: a target step of --step rad/s with --dwell seconds of dwell per step, so each step
     reaches steady state before it is judged.
  4. Threshold per wheel: the first step at which |dq| exceeds --onset-dq.
  5. Measured solver torque is recorded per step so the torque at break-away is available.

Nothing is tuned and no gate is changed.
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
WHEEL_KD, ACTION_CLIP, WHEEL_SCALE, WHEEL_TAU_CAP = 6.0, 6.0, 16.0, 13.5
DT = 0.02
WHEELS = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


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
    ap.add_argument("--baseline-s", type=float, default=1.0)
    ap.add_argument("--baseline-dq", type=float, default=0.02)
    ap.add_argument("--step", type=float, default=0.02, help="target increment per step, rad/s")
    ap.add_argument("--dwell", type=float, default=0.5, help="dwell per step, s")
    ap.add_argument("--tgt-max", type=float, default=0.40)
    ap.add_argument("--onset-dq", type=float, default=0.05)
    # The VOID_BASELINE from the first redesigned run showed the robot is NOT still when only the
    # wheels are forced to zero: the policy keeps moving the legs, and the resulting body motion
    # drags the wheels (creep up to 0.35 rad/s).  So the legs must be frozen at the fixed stand as
    # well, otherwise the wheels can never be measured from rest.
    ap.add_argument("--freeze-legs", type=int, default=1,
                    help="1 = hold the leg action at 0 (fixed stand) during baseline and ramp")
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
            names = list(robot.joint_names)
            widx = [names.index(n) for n in WHEELS]
            print("WHEEL_JOINTS %s" % json.dumps([names[i] for i in widx]), flush=True)

            ppo = PPOCfg(); ppo.seed, ppo.device = args.seed, args.device
            ppo.max_iterations = 1
            ppo = handle_dep(ppo, check_v())
            runner = create_runner(wrapped, ppo, log_dir=None)
            runner.load(str(args.ckpt), map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)

            def measured_tau():
                if not hasattr(view, "get_dof_projected_joint_forces"):
                    return None
                f = view.get_dof_projected_joint_forces()
                t = f.torch if hasattr(torch, "no_grad") and hasattr(f, "torch") else torch.as_tensor(f, device=env.device)
                return t[0][widx].detach().cpu().numpy().astype(float)

            def step_with(tgt_all, obs):
                """Advance one policy step with the wheel targets forced to tgt_all.

                With --freeze-legs the leg action is held at 0, which is the fixed stand (the leg
                action has no offset, so 0 means 'hold the default stance').
                """
                with torch.inference_mode():
                    a = policy(obs)
                a = a.clone()
                if args.freeze_legs:
                    a[0, :12] = 0.0
                for j in range(4):
                    a[0, 12 + j] = float(tgt_all) / WHEEL_SCALE
                obs, _r, _d, _i = wrapped.step(a)
                dq = robot.data.joint_vel.torch[0][widx].detach().cpu().numpy().copy()
                tau = measured_tau()
                return obs, dq, tau, bool(_d[0])

            env.seed(args.seed); env.reset()
            cmdterm.vel_command_b[:] = 0.0
            obs = wrapped.get_observations()
            for _ in range(int(args.settle_s / DT)):
                with torch.inference_mode():
                    a = policy(obs)
                obs, _r, _d, _i = wrapped.step(a)

            # ---- baseline gate -------------------------------------------------
            base = []
            for _ in range(int(args.baseline_s / DT)):
                obs, dq, tau, term = step_with(0.0, obs)
                base.append(dq)
                if term:
                    break
            base = np.array(base)
            base_absmax = float(np.abs(base).max())
            baseline_ok = base_absmax < args.baseline_dq
            print("BASELINE absmax_dq=%.4f rad/s gate=%.3f -> %s"
                  % (base_absmax, args.baseline_dq, "OK" if baseline_ok else "VOID"), flush=True)

            rep = {"checkpoint": str(args.ckpt), "seed": args.seed, "mode": "common_mode_ramp",
                   "baseline_absmax_dq": base_absmax, "baseline_gate": args.baseline_dq,
                   "baseline_ok": baseline_ok, "step_rad_s": args.step, "dwell_s": args.dwell,
                   "tgt_max_rad_s": args.tgt_max, "onset_dq": args.onset_dq,
                   "wheel_kd": WHEEL_KD, "wheel_tau_cap_nm": WHEEL_TAU_CAP,
                   "freeze_legs": int(args.freeze_legs), "steps": [], "per_wheel": {}}

            dwell_n = max(1, int(args.dwell / DT))
            n_steps = int(round(args.tgt_max / args.step))
            traces = {w: [] for w in WHEELS}
            terms = 0
            for s in range(1, n_steps + 1):
                tgt = s * args.step
                ds, ts = [], []
                for _ in range(dwell_n):
                    obs, dq, tau, term = step_with(tgt, obs)
                    ds.append(dq)
                    if tau is not None:
                        ts.append(tau)
                    terms += int(term)
                    if term:
                        break
                dsm = np.array(ds)
                # steady-state = second half of the dwell
                half = max(1, len(dsm) // 2)
                dq_ss = dsm[half:].mean(axis=0)
                tau_ss = (np.array(ts)[half:].mean(axis=0) if ts else None)
                rec = {"tgt": float(tgt),
                       "dq_ss": [float(v) for v in dq_ss],
                       "tau_ss": ([float(v) for v in tau_ss] if tau_ss is not None else None),
                       "dq_absmax_within_step": float(np.abs(dsm).max())}
                rep["steps"].append(rec)
                for j, w in enumerate(WHEELS):
                    traces[w].append((float(tgt), float(dq_ss[j]),
                                      float(tau_ss[j]) if tau_ss is not None else None))
                print("  tgt=%.2f dq_ss=%s tau_ss=%s"
                      % (tgt, " ".join("%+.3f" % v for v in dq_ss),
                         (" ".join("%+.2f" % v for v in tau_ss) if tau_ss is not None else "-")),
                      flush=True)

            for w in WHEELS:
                tr = traces[w]
                onset = next(((t, d, ta) for (t, d, ta) in tr if abs(d) > args.onset_dq), None)
                rep["per_wheel"][w] = {
                    "onset_tgt_rad_s": onset[0] if onset else None,
                    "onset_dq_rad_s": onset[1] if onset else None,
                    "onset_tau_nm": onset[2] if onset else None,
                    "tracks_at_tgt_max": (abs(tr[-1][1]) > args.onset_dq) if tr else None,
                    "dq_at_tgt_max": tr[-1][1] if tr else None,
                }
            rep["terms"] = terms
            rep["verdict"] = ("VALID" if baseline_ok and not terms and
                              any(v["onset_tgt_rad_s"] is not None for v in rep["per_wheel"].values())
                              else ("VOID_BASELINE" if not baseline_ok else
                                    ("VOID_TERMINATED" if terms else "NO_ONSET_WITHIN_RAMP")))
            print()
            for w in WHEELS:
                v = rep["per_wheel"][w]
                print("  %s onset=%s rad/s tau=%s Nm | at tgt_max dq=%s"
                      % (w,
                         ("%.2f" % v["onset_tgt_rad_s"]) if v["onset_tgt_rad_s"] is not None else "none",
                         ("%+.2f" % v["onset_tau_nm"]) if v["onset_tau_nm"] is not None else "-",
                         ("%+.3f" % v["dq_at_tgt_max"]) if v["dq_at_tgt_max"] is not None else "-"))
            print("VERDICT %s" % rep["verdict"], flush=True)
            (args.out / "wheel_floor2.json").write_text(json.dumps(rep, indent=1, default=str))
            print("WHEEL_FLOOR2_DONE", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
