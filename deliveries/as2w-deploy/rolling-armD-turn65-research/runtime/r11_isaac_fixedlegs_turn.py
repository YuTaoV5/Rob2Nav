#!/usr/bin/env python3
"""Operator item 2: validate a FIXED standard leg reference separately from the actor's leg pose.

The MuJoCo work showed the pure-turn failures split into command generation and support loss, and the
operator wants the leg axis validated on its own: ramp smoothly from the fixed stand to the standard
rolling joint reference, keep the leg PD active, then drive the SAME wheel command sequence and compare
against the actor-driven leg pose.

Arms:
  actor       legs from the policy                (baseline)
  fixed       legs held at the standard reference  (leg action forced to 0, which is DEFAULT + PD)
  fixed_diff  fixed legs plus the bounded, command-following differential target

Leg transition: the leg action is blended from the actor's value to the fixed reference with a cosine
ramp over --ramp-s, so nothing snaps.

Support is measured on the WHEEL BODIES this time (the previous replay mapped wheel contacts to calf
joints, which made support read 0.000 - that defect is fixed here).

Reports per case: tail-3 s yaw track, support fraction, upright minimum, terminations, the measured
tail differential and the leg deviation from the standard reference in radians.
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
DT = 0.02
TAIL = 150
LOAD_N = 5.0
S_WHEEL = np.array([-1.0, 1.0, -1.0, 1.0])
ARMS = ["actor", "fixed", "fixed_diff"]
CASES = [("turn_left_0.05", 0.0, 0.05), ("turn_right_0.05", 0.0, -0.05),
         ("turn_left_0.10", 0.0, 0.10), ("turn_right_0.10", 0.0, -0.10),
         ("arc_right_0.25", 0.25, -0.05), ("arc_left_0.25", 0.25, 0.05)]


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
    ap.add_argument("--settle-s", type=float, default=2.0)
    ap.add_argument("--ramp-s", type=float, default=1.0)
    ap.add_argument("--case-s", type=float, default=12.0)
    ap.add_argument("--diff-gain", type=float, default=2.0)
    ap.add_argument("--diff-cap", type=float, default=0.5)
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
            widx = [names.index(n) for n in
                    ("FL_foot", "FR_foot", "RL_foot", "RR_foot")]
            # CORRECT support indexing: the wheel BODIES, not the calf joints.  The previous
            # replay mapped to calf joints, which made support read 0.000.
            wbid = [bodies.index(n) for n in
                    ("FL_foot", "FR_foot", "RL_foot", "RR_foot")]
            print("WHEEL_JOINTS %s" % json.dumps([names[i] for i in widx]), flush=True)
            print("WHEEL_BODIES %s" % json.dumps([bodies[i] for i in wbid]), flush=True)

            ppo = PPOCfg(); ppo.seed, ppo.device = args.seed, args.device
            ppo.max_iterations = 1
            ppo = handle_dep(ppo, check_v())
            runner = create_runner(wrapped, ppo, log_dir=None)
            runner.load(str(args.ckpt), map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)

            def run(arm, vx, wz):
                env.seed(args.seed); env.reset()
                cmdterm.vel_command_b[:] = 0.0
                obs = wrapped.get_observations()
                for _ in range(int(args.settle_s / DT)):
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, _r, _d, _i = wrapped.step(a)
                rows = []
                n_ramp = max(1, int(args.ramp_s / DT))
                n = int(args.case_s / DT)
                for k in range(n):
                    cmdterm.vel_command_b[:] = torch.tensor(
                        [[vx, 0.0, wz]], dtype=torch.float32, device=env.device)
                    with torch.inference_mode():
                        a = policy(obs)
                    a = a.clone()
                    if arm in ("fixed", "fixed_diff"):
                        # cosine blend from the actor's leg action to 0 (= hold DEFAULT + PD)
                        w = 1.0
                        if k < n_ramp:
                            w = 0.5 * (1.0 + np.cos(np.pi * k / n_ramp))
                        a[0, :12] = a[0, :12] * (1.0 - w)
                    if arm == "fixed_diff":
                        mag = float(np.clip(args.diff_gain * wz, -args.diff_cap, args.diff_cap))
                        a[0, 12:16] += torch.tensor(S_WHEEL * mag / WHEEL_SCALE,
                                                    dtype=a.dtype, device=a.device)
                    obs, _r, _d, _i = wrapped.step(a)
                    cf = env.scene["contact_forces"].data.net_forces_w.torch[0][wbid]
                    load = torch.linalg.norm(cf, dim=1)
                    qa = env.action_manager.action[0]
                    rows.append({
                        "yaw": float(robot.data.root_ang_vel_b.torch[0, 2]),
                        "up": float(-robot.data.projected_gravity_b.torch[0][2]),
                        "sup": bool((load > LOAD_N).all()),
                        "app": (qa[12:16] * WHEEL_SCALE).detach().cpu().numpy().copy(),
                        "mea": robot.data.joint_vel.torch[0][widx].detach().cpu().numpy().copy(),
                        "term": bool(_d[0]),
                        # deviation of the actual joint angle from the default reference
                        "leg_dev": float(torch.abs(robot.data.joint_pos.torch[0][:12]
                                                   - robot.data.default_joint_pos.torch[0][:12]).max()),
                    })
                    if rows[-1]["term"]:
                        break
                tl = slice(max(0, len(rows) - TAIL), len(rows))
                tgt = np.array([r["app"] for r in rows]); mea = np.array([r["mea"] for r in rows])
                dtg, dme = float(np.mean(d4(tgt)[tl])), float(np.mean(d4(mea)[tl]))
                return {
                    "arm": arm, "vx": vx, "wz": wz, "n": len(rows),
                    "yaw_tail": float(np.mean([r["yaw"] for r in rows[tl]])),
                    "yaw_track": (float(np.mean([r["yaw"] for r in rows[tl]]) / wz) if abs(wz) > 1e-9 else None),
                    "support_tail": float(np.mean([r["sup"] for r in rows[tl]])),
                    "support_all": float(np.mean([r["sup"] for r in rows])),
                    "upright_min": float(np.min([r["up"] for r in rows])),
                    "ddq_target_tail": dtg, "ddq_measured_tail": dme,
                    "exec": (abs(dme) / abs(dtg)) if abs(dtg) > 1e-9 else None,
                    "leg_dev_max_rad": float(np.max([r["leg_dev"] for r in rows])),
                    "terminations": int(sum(r["term"] for r in rows)),
                }

            rep = {"checkpoint": str(args.ckpt), "seed": args.seed, "ramp_s": args.ramp_s,
                   "diff_gain": args.diff_gain, "diff_cap": args.diff_cap,
                   "note": "arms differ only in the leg source (and fixed_diff adds the bounded differential target); wheel command sequence otherwise identical",
                   "support_indexing": "wheel BODIES (fixes the calf-joint defect from the earlier replay)",
                   "results": []}
            for arm in ARMS:
                for case, vx, wz in CASES:
                    r = run(arm, vx, wz)
                    r["case"] = case
                    rep["results"].append(r)
                    print("%-11s %-16s yaw=%+.2f sup_t=%.3f up=%.3f exec=%.2f legdev=%.3f term=%d"
                          % (arm, case, r["yaw_track"] or 0, r["support_tail"],
                             r["upright_min"], r["exec"] or 0, r["leg_dev_max_rad"],
                             r["terminations"]), flush=True)
            (args.out / "fixedlegs_turn.json").write_text(json.dumps(rep, indent=1, default=str))
            print("FIXEDLEGS_TURN_DONE", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
