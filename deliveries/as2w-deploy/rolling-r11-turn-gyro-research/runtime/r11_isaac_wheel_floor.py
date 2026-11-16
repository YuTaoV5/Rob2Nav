#!/usr/bin/env python3
"""Isaac: measure the wheel break-away (rolling-resistance / static-friction) floor.

The previous round established that at near-zero speed the kd law puts out 1-4.4 Nm while the wheel
barely rotates.  The next question is quantitative: HOW MUCH torque or commanded speed does a wheel
actually need before it starts turning on the ground?

Method: settle under the policy at zero command, then bypass the policy's wheel part and ramp one
wheel's target speed slowly from 0 upward while the other three are held at 0.  The break-away point
is the first sample where |dq| exceeds a small threshold.  Record at that moment:
  * commanded target speed (rad/s)
  * measured wheel speed (rad/s)
  * measured solver-projected torque (Nm)
  * reconstructed torque KD*(target-actual) for comparison

Doing this per wheel gives the per-wheel floor and exposes left/right differences.  The legs keep
following the policy so the stance stays the one the policy produces.

Nothing is gated and nothing is tuned: this is a measurement, run before any decision about wheel
gain or compensation, as the operator instructed.
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
    ap.add_argument("--ramp-s", type=float, default=6.0)
    ap.add_argument("--tgt-max", type=float, default=2.0, help="ramp the target up to this rad/s")
    ap.add_argument("--onset-dq", type=float, default=0.05, help="|dq| above this counts as turning")
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
            print("HAS_PROJECTED %s" % hasattr(view, "get_dof_projected_joint_forces"), flush=True)

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
                t = f.torch if hasattr(f, "torch") else torch.as_tensor(f, device=env.device)
                return t[0][widx].detach().cpu().numpy().astype(float)

            def settle():
                env.seed(args.seed); env.reset()
                cmdterm.vel_command_b[:] = 0.0
                obs = wrapped.get_observations()
                for _ in range(int(args.settle_s / DT)):
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, _r, _d, _i = wrapped.step(a)
                return obs

            def ramp(drive_i):
                """Ramp wheel drive_i's target; run through the env so the legs keep following policy."""
                obs = settle()
                rows = []
                n = int(args.ramp_s / DT)
                for k in range(n):
                    tgt_cmd = args.tgt_max * (k + 1) / n
                    with torch.inference_mode():
                        a = policy(obs)
                    a = a.clone()
                    a[0, 12 + drive_i] = float(tgt_cmd) / WHEEL_SCALE
                    obs, _r, _d, _i = wrapped.step(a)
                    dq = robot.data.joint_vel.torch[0][widx].detach().cpu().numpy().copy()
                    tau = measured_tau()
                    rec = WHEEL_KD * (np.array([0.0] * 4) - dq)
                    rec[drive_i] = WHEEL_KD * (tgt_cmd - dq[drive_i])
                    ta = tau[drive_i] if tau is not None else None
                    rows.append({"t": k * DT, "tgt": float(tgt_cmd),
                                 "dq_drive": float(dq[drive_i]),
                                 "dq_all": [float(v) for v in dq],
                                 "tau_drive": float(ta) if ta is not None else None,
                                 "recon_drive": float(rec[drive_i])})
                    if bool(_d[0]):
                        rows.append({"term": True, "t": k * DT})
                        break
                return rows

            rep = {"checkpoint": str(args.ckpt), "seed": args.seed,
                   "ramp_s": args.ramp_s, "tgt_max_rad_s": args.tgt_max,
                   "onset_dq_rad_s": args.onset_dq,
                   "wheel_kd": WHEEL_KD, "wheel_tau_cap_nm": WHEEL_TAU_CAP,
                   "note": "one wheel ramped at a time with the other three held at a zero target; legs keep following the policy",
                   "per_wheel": {}}
            for i, w in enumerate(WHEELS):
                rows = ramp(i)
                onset = next((r for r in rows if "dq_drive" in r and abs(r["dq_drive"]) > args.onset_dq), None)
                term = any(r.get("term") for r in rows)
                # torque at the last sample before onset = what it took to break away
                pre = None
                if onset is not None:
                    idx = next(k for k, r in enumerate(rows) if r is onset)
                    pre = rows[max(0, idx - 1)]
                rep["per_wheel"][w] = {
                    "onset_tgt_rad_s": onset["tgt"] if onset else None,
                    "onset_dq_rad_s": onset["dq_drive"] if onset else None,
                    "onset_tau_nm": onset["tau_drive"] if onset else None,
                    "pre_onset_tau_nm": pre["tau_drive"] if pre else None,
                    "pre_onset_tgt_rad_s": pre["tgt"] if pre else None,
                    "max_abs_tau_nm": max((abs(r["tau_drive"]) for r in rows if r.get("tau_drive") is not None), default=None),
                    "terminated": term,
                    "trace": rows,
                }
                print("  %s onset_tgt=%s rad/s  onset_tau=%s Nm  pre_onset_tau=%s  max_tau=%s  term=%s"
                      % (w,
                         ("%.3f" % onset["tgt"]) if onset else "not within ramp",
                         ("%+.2f" % onset["tau_drive"]) if onset and onset["tau_drive"] is not None else "-",
                         ("%+.2f" % pre["tau_drive"]) if pre and pre["tau_drive"] is not None else "-",
                         ("%.2f" % rep["per_wheel"][w]["max_abs_tau_nm"]) if rep["per_wheel"][w]["max_abs_tau_nm"] is not None else "-",
                         term), flush=True)
            (args.out / "wheel_floor.json").write_text(json.dumps(rep, indent=1, default=str))
            print("WHEEL_FLOOR_DONE", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
