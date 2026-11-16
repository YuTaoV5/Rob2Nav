#!/usr/bin/env python3
"""Continuous joystick sequence regression, both directions, with a correct stand reference.

Timeline per direction (one continuous trajectory, no reset between stages):

    0 - 3 s    fixed stand, zero command        <- the LAST 1 s of this is the stand reference
    3 - 4 s    ramp to +-0.25 m/s
    4 - 9 s    hold the commanded speed
    9 - 19 s   release: command to zero, decelerate, come to rest and hold (10 s)
   19 - 22 s   restore the standing posture

The stand reference is the posture the policy ACTUALLY holds at zero command in this same episode,
not default_joint_pos.  An earlier version compared against default_joint_pos with a 0.20 rad
tolerance and reported a false failure: the policy's ordinary rolling stance already sits 0.30-0.52
rad away from the URDF default, so that tolerance was unreachable by construction.

Judged on the same trajectory, per direction:
  decelerate / come to rest - stop time and distance, tail |vx| and |wz| against the frozen 0.02 gates
  restore                   - final leg posture against the episode's own zero-command posture
  safety                    - no termination, four-wheel support, upright, no non-wheel contact
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
UPRIGHT_MIN, SUPPORT_MIN = 0.85, 0.95
STAND_TOL = 0.10          # rad, against the episode's OWN zero-command posture
STAND_REF_WINDOW = 50     # last 1 s of the stand-in stage
STAGES = [("stand_in", 3.0), ("ramp", 1.0), ("hold", 5.0), ("release", 10.0), ("stand_out", 3.0)]


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
    ap.add_argument("--direction", choices=["rev", "fwd", "both"], default="both")
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=1)
    add_launcher_args(ap)
    args = ap.parse_args()
    for k in ("usd", "urdf", "ckpt"):
        setattr(args, k, getattr(args, k).resolve(strict=True))
    args.out.mkdir(parents=True, exist_ok=True)
    args.physics = "isaacsim_physx"

    dirs = {"rev": [-0.25], "fwd": [0.25], "both": [-0.25, 0.25]}[args.direction]

    with launch_simulation(SimulationCfg(device=args.device, dt=.005), args) as physics:
        env, Wrapper, PPOCfg, check_v, create_runner, handle_dep = build(args, physics)
        try:
            wrapped = Wrapper(env, clip_actions=ACTION_CLIP)
            robot = env.scene["robot"]
            cmdterm = env.command_manager.get_term("base_velocity")
            names = list(robot.joint_names); bodies = list(robot.body_names)
            wbid = [bodies.index(n) for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")]
            leg_ids = list(range(12))

            ppo = PPOCfg(); ppo.seed, ppo.device = args.seed, args.device
            ppo.max_iterations = 1
            ppo = handle_dep(ppo, check_v())
            runner = create_runner(wrapped, ppo, log_dir=None)
            runner.load(str(args.ckpt), map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)

            def legs():
                return robot.data.joint_pos.torch[0][leg_ids].detach().cpu().numpy().copy()

            def run(vx_target):
                env.seed(args.seed); env.reset()
                cmdterm.vel_command_b[:] = 0.0
                obs = wrapped.get_observations()
                rows, term = [], 0
                stand_ref = None
                for stage, secs in STAGES:
                    n = int(secs / DT)
                    for k in range(n):
                        vx = 0.0
                        if stage == "ramp":
                            vx = vx_target * (k + 1) / n            # smooth ramp
                        elif stage == "hold":
                            vx = vx_target
                        cmdterm.vel_command_b[:] = torch.tensor(
                            [[vx, 0.0, 0.0]], dtype=torch.float32, device=env.device)
                        with torch.inference_mode():
                            a = policy(obs)
                        obs, _r, _d, _i = wrapped.step(a)
                        cf = env.scene["contact_forces"].data.net_forces_w.torch[0][wbid]
                        load = torch.linalg.norm(cf, dim=1)
                        rows.append({
                            "stage": stage, "t": len(rows) * DT,
                            "vx": float(robot.data.root_lin_vel_b.torch[0, 0]),
                            "wz": float(robot.data.root_ang_vel_b.torch[0, 2]),
                            "up": float(-robot.data.projected_gravity_b.torch[0][2]),
                            "sup": bool((load > LOAD_N).all()),
                            "legs": legs(),
                            "term": bool(_d[0]),
                        })
                        term += int(bool(_d[0]))
                        # the stand reference is the posture held at zero command, late in stage 0
                        if stage == "stand_in" and k >= n - STAND_REF_WINDOW:
                            stand_ref = (legs() if stand_ref is None
                                         else 0.5 * (stand_ref + legs()))
                return rows, term, stand_ref

            rep = {"checkpoint": str(args.ckpt), "seed": args.seed,
                   "gates": {"stop_vx": STOP_VX, "stop_wz": STOP_WZ, "upright": UPRIGHT_MIN,
                              "support": SUPPORT_MIN, "stand_tol_rad": STAND_TOL},
                   "stand_reference": "episode's own zero-command posture (last 1 s of stand_in), NOT default_joint_pos",
                   "note": "earlier version used default_joint_pos with 0.20 rad and produced a false failure because the policy stance sits 0.30-0.52 rad from that default",
                   "directions": {}}
            for vx_target in dirs:
                rows, term, stand_ref = run(vx_target)
                rel = [r for r in rows if r["stage"] == "release"]
                stop = next((i for i, r in enumerate(rel) if abs(r["vx"]) < STOP_VX), None)
                dist = float(sum(abs(r["vx"]) for r in rel[:stop + 1]) * DT) if stop is not None else None
                tail = rel[-150:] if len(rel) >= 150 else rel
                out_stage = [r for r in rows if r["stage"] == "stand_out"][-150:]
                ref = stand_ref if stand_ref is not None else np.zeros(12)
                err = max(float(np.abs(r["legs"] - ref).max()) for r in out_stage) if out_stage else None
                d = {
                    "cmd_vx": vx_target,
                    "stop_time_s": (stop * DT) if stop is not None else None,
                    "stop_distance_m": dist,
                    "tail_vx": float(np.mean([r["vx"] for r in tail])),
                    "tail_wz": float(np.mean([r["wz"] for r in tail])),
                    "tail_support": float(np.mean([r["sup"] for r in tail])),
                    "upright_min": float(min(r["up"] for r in rows)),
                    "terminations": term,
                    "final_leg_err_vs_stand_ref": err,
                    "final_vx": float(np.mean([r["vx"] for r in out_stage])) if out_stage else None,
                }
                d["pass"] = bool(
                    term == 0 and d["upright_min"] >= UPRIGHT_MIN
                    and d["tail_support"] >= SUPPORT_MIN
                    and abs(d["tail_vx"]) < STOP_VX and abs(d["tail_wz"]) < STOP_WZ
                    and d["stop_time_s"] is not None
                    and err is not None and err <= STAND_TOL
                )
                rep["directions"]["fwd" if vx_target > 0 else "rev"] = d
                print("JOYSTICK_SEQ dir=%s pass=%s stop_t=%s s stop_d=%s m tail_vx=%+.4f tail_wz=%+.4f "
                      "sup=%.3f up_min=%.3f leg_err_vs_stand=%.3f term=%d"
                      % ("fwd" if vx_target > 0 else "rev", d["pass"], d["stop_time_s"], d["stop_distance_m"],
                         d["tail_vx"], d["tail_wz"], d["tail_support"], d["upright_min"], err, term), flush=True)
            rep["all_pass"] = all(v["pass"] for v in rep["directions"].values())
            (args.out / "joystick_seq.json").write_text(json.dumps(rep, indent=1, default=str))
            print("JOYSTICK_SEQ_DONE all_pass=%s" % rep["all_pass"], flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
