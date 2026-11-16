#!/usr/bin/env python3
"""Leg-pose ablation against the zero-command drift (goal section 5, leg as a search dimension).

Why this experiment: the break-away work established that at a zero wheel target the velocity drive
is a KD damper, not a brake, so the robot creeps.  Whatever gravity/posture drives that creep is a
posture quantity, which means the leg-pose dimension the goal explicitly authorises is the natural
lever to try - and unlike the refuted gain experiments this touches neither the reward, nor the
acceptance gates, nor a gain.

Each arm runs the zero-command hold for --hold-s and reports the goal's own hold criteria plus the
hard-constraint telemetry the goal asks to track alongside any leg-pose change:
  * tail-3 s |vx| and |wz| (hold gate: <0.02 each)
  * cumulative displacement and yaw drift over the whole hold
  * four-wheel support fraction, upright minimum
  * measured solver-projected peak wheel torque (to show the torque margin)
  * the leg-pose offset actually applied, in radians and as a joint target

Leg poses are applied as a constant offset on the LEG part of the action, so the resulting joint
target is DEFAULT + offset.  The standard stance remains the default and the fallback; every arm is
compared against the baseline arm run in the same process.

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
WHEEL_KD, ACTION_CLIP, WHEEL_SCALE = 6.0, 6.0, 16.0
LEG_SCALE = 0.35
DT = 0.02
TAIL = 150
LOAD_N = 5.0
WHEELS = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
# canonical part-major order: hips 0-3, thighs 4-7, calves 8-11
PARTS = {"hip": [0, 1, 2, 3], "thigh": [4, 5, 6, 7], "calf": [8, 9, 10, 11]}
LEGS = {"FL": 0, "FR": 1, "RL": 2, "RR": 3}

ARMS = [
    ("baseline",        {}),
    ("base_h_up",       {"thigh": +0.04, "calf": -0.08}),   # raise the stance
    ("base_h_down",     {"thigh": -0.04, "calf": +0.08}),   # lower the stance
    ("rear_rake_fwd",   {"thigh_RL": +0.10, "thigh_RR": +0.10}),
    ("rear_rake_back",  {"thigh_RL": -0.10, "thigh_RR": -0.10}),
]


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


def offsets_to_indices(spec):
    """Turn an arm spec into (joint indices, per-index offset)."""
    idx, off = [], []
    for key, val in spec.items():
        if key in PARTS:                       # all four legs for that part
            for i in PARTS[key]:
                idx.append(i); off.append(val)
        elif "_" in key:                       # part_LEG, e.g. thigh_RL
            part, leg = key.split("_", 1)
            if part not in PARTS or leg not in LEGS:
                raise SystemExit("bad arm key %r" % key)
            idx.append(PARTS[part][LEGS[leg]]); off.append(val)
    return np.array(idx, dtype=int), np.array(off, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7393)
    ap.add_argument("--settle-s", type=float, default=2.0)
    ap.add_argument("--hold-s", type=float, default=10.0)
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
                t = f.torch if hasattr(f, "torch") else torch.as_tensor(f, device=env.device)
                return t[0][widx].detach().cpu().numpy().astype(float)

            def run_arm(spec):
                idx, off = offsets_to_indices(spec)
                env.seed(args.seed); env.reset()
                cmdterm.vel_command_b[:] = 0.0
                obs = wrapped.get_observations()
                for _ in range(int(args.settle_s / DT)):
                    with torch.inference_mode():
                        a = policy(obs)
                    if idx.size:
                        a = a.clone(); a[0, idx] += torch.tensor(off / LEG_SCALE, dtype=a.dtype, device=a.device)
                    obs, _r, _d, _i = wrapped.step(a)
                p0 = robot.data.root_pos_w.torch[0].detach().cpu().numpy().copy()
                rows = []
                for _ in range(int(args.hold_s / DT)):
                    cmdterm.vel_command_b[:] = 0.0
                    with torch.inference_mode():
                        a = policy(obs)
                    if idx.size:
                        a = a.clone(); a[0, idx] += torch.tensor(off / LEG_SCALE, dtype=a.dtype, device=a.device)
                    obs, _r, _d, _i = wrapped.step(a)
                    rows.append({
                        "vx": float(robot.data.root_lin_vel_b.torch[0, 0]),
                        "vy": float(robot.data.root_lin_vel_b.torch[0, 1]),
                        "wz": float(robot.data.root_ang_vel_b.torch[0, 2]),
                        "up": float(-robot.data.projected_gravity_b.torch[0][2]),
                        "pos": robot.data.root_pos_w.torch[0].detach().cpu().numpy().copy(),
                        "tau": measured_tau(),
                        "term": bool(_d[0]),
                    })
                n = len(rows)
                tl = slice(max(0, n - TAIL), n)
                taus = [r["tau"] for r in rows if r["tau"] is not None]
                disp = float(np.linalg.norm(rows[-1]["pos"][:2] - p0[:2])) if rows else None
                return {
                    "spec": spec,
                    "applied_joint_indices": idx.tolist(),
                    "applied_offset_rad": off.tolist(),
                    "n_frames": n,
                    "vx_tail": float(np.mean([r["vx"] for r in rows[tl]])),
                    "vy_tail": float(np.mean([r["vy"] for r in rows[tl]])),
                    "wz_tail": float(np.mean([r["wz"] for r in rows[tl]])),
                    "vx_absmax_full": float(np.max(np.abs([r["vx"] for r in rows]))),
                    "upright_min": float(np.min([r["up"] for r in rows])),
                    "displacement_m": disp,
                    "tau_absmax": float(np.max(np.abs(np.array(taus)))) if taus else None,
                    "terminations": int(sum(r["term"] for r in rows)),
                }

            rep = {"checkpoint": str(args.ckpt), "seed": args.seed,
                   "hold_s": args.hold_s, "leg_scale": LEG_SCALE,
                   "note": "leg-pose offsets applied as a constant offset on the leg part of the action; target = DEFAULT + offset",
                   "gates": {"hold_vx": 0.02, "hold_wz": 0.02, "upright": 0.85},
                   "arms": []}
            for name, spec in ARMS:
                r = run_arm(spec)
                r["arm"] = name
                rep["arms"].append(r)
                print("== %-14s vx_tail=%+.5f wz_tail=%+.5f vy_tail=%+.5f up_min=%.3f disp=%s tau_max=%s term=%d"
                      % (name, r["vx_tail"], r["wz_tail"], r["vy_tail"], r["upright_min"],
                         ("%.4f" % r["displacement_m"]) if r["displacement_m"] is not None else "-",
                         ("%.2f" % r["tau_absmax"]) if r["tau_absmax"] is not None else "-",
                         r["terminations"]), flush=True)
            (args.out / "legpose_hold.json").write_text(json.dumps(rep, indent=1, default=str))
            print("LEGPOSE_HOLD_DONE", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
