#!/usr/bin/env python3
"""Does ground/wheel friction explain Isaac's missing pure-turn yaw?  Single-variable sweep.

Three earlier suspects are already eliminated: wheel differential delivery (0.89 delivery still gives
0.06 yaw), leg pose (freezing the legs cuts the arcs by 74-97 percent) and torque saturation (measured
peak 32 percent of the cap).  Pure spin-in-place with non-steerable wheels REQUIRES lateral sliding, so
the friction that governs lateral slip is the natural remaining suspect, and the goal asks for nominal
friction to be unified before cross-simulator comparison.

MuJoCo's ground geom uses friction=".62 .005 .0001" (sliding 0.62).  The Isaac friction event only
activates in the friction/combined/expanded profiles, NOT in the pose-delay profile used for
acceptance, so what Isaac actually ran with has never been checked.

This probe first PRINTS the friction it can read, then pins the wheel material to fixed values and
measures pure-turn and arc yaw.  Only friction changes between arms.
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
TAIL = 150
LOAD_N = 5.0
CASES = [("turn_left_0.05", 0.0, 0.05), ("turn_right_0.05", 0.0, -0.05),
         ("turn_left_0.10", 0.0, 0.10), ("turn_right_0.10", 0.0, -0.10),
         ("arc_right_0.25", 0.25, -0.05), ("arc_left_0.25", 0.25, 0.05)]
FRICTIONS = [None, 1.2, 0.62, 0.35]      # None = leave Isaac as shipped


def build(args, physics):
    from isaaclab.managers import EventTermCfg as EvCfg, SceneEntityCfg
    import isaaclab.envs.mdp as mdp
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

    if args.friction is not None:
        f = float(args.friction)
        # fixed, not randomised: same value for static and dynamic so the arm is one variable
        cfg.events.as2w_friction_fix = EvCfg(
            func=mdp.randomize_rigid_body_material, mode="startup", params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "static_friction_range": (f, f), "dynamic_friction_range": (f, f),
                "restitution_range": (0., 0.), "num_buckets": 1,
            })
    env = RobustEnvironment(cfg)
    return env, RslRlVecEnvWrapper, UnitreeGo2FlatPPORunnerCfg, check_rsl_rl_version, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--friction", type=float, default=None)
    ap.add_argument("--seed", type=int, default=7393)
    ap.add_argument("--settle-s", type=float, default=2.0)
    ap.add_argument("--case-s", type=float, default=12.0)
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
            names = list(robot.joint_names); bodies = list(robot.body_names)
            widx = [names.index(n) for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")]
            wbid = [bodies.index(n) for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")]

            # --- report whatever friction is actually in effect -------------------------------
            info = {"requested_wheel_friction": args.friction, "read": {}}
            try:
                view = robot.root_physx_view
                for meth in ("get_material_properties",):
                    if hasattr(view, meth):
                        mp = getattr(view, meth)()
                        t = mp.torch if hasattr(mp, "torch") else torch.as_tensor(mp)
                        arr = t.detach().cpu().numpy()
                        info["read"]["material_properties_shape"] = list(arr.shape)
                        info["read"]["wheel_rows"] = [arr[0][i].tolist() if arr.ndim >= 2 else None
                                                     for i in wbid]
            except Exception as exc:
                info["read"]["error"] = "%s: %s" % (type(exc).__name__, exc)
            print("FRICTION_INFO %s" % json.dumps(info)[:500], flush=True)

            ppo = PPOCfg(); ppo.seed, ppo.device = args.seed, args.device
            ppo.max_iterations = 1
            ppo = handle_dep(ppo, check_v())
            runner = create_runner(wrapped, ppo, log_dir=None)
            runner.load(str(args.ckpt), map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)

            def run(vx, wz):
                env.seed(args.seed); env.reset()
                cmdterm.vel_command_b[:] = 0.0
                obs = wrapped.get_observations()
                for _ in range(int(args.settle_s / DT)):
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, _r, _d, _i = wrapped.step(a)
                yaw, up, sup, term = [], [], [], 0
                for _ in range(int(args.case_s / DT)):
                    cmdterm.vel_command_b[:] = torch.tensor([[vx, 0.0, wz]],
                                                            dtype=torch.float32, device=env.device)
                    with torch.inference_mode():
                        a = policy(obs)
                    obs, _r, _d, _i = wrapped.step(a)
                    cf = env.scene["contact_forces"].data.net_forces_w.torch[0][wbid]
                    load = torch.linalg.norm(cf, dim=1)
                    yaw.append(float(robot.data.root_ang_vel_b.torch[0, 2]))
                    up.append(float(-robot.data.projected_gravity_b.torch[0][2]))
                    sup.append(bool((load > LOAD_N).all()))
                    term += int(bool(_d[0]))
                tl = slice(max(0, len(yaw) - TAIL), len(yaw))
                return {"yaw_tail": float(np.mean(yaw[tl])),
                        "yaw_track": (float(np.mean(yaw[tl]) / wz) if abs(wz) > 1e-9 else None),
                        "support_tail": float(np.mean(sup[tl])),
                        "upright_min": float(np.min(up)), "terminations": term}

            rep = {"checkpoint": str(args.ckpt), "seed": args.seed, "friction": args.friction,
                   "note": "only the wheel material friction changes between runs", "info": info,
                   "cases": {}}
            for case, vx, wz in CASES:
                r = run(vx, wz); r["vx"], r["wz"] = vx, wz
                rep["cases"][case] = r
                print("== f=%-6s %-16s yaw_track=%+.2f sup_t=%.3f up=%.3f term=%d"
                      % (str(args.friction), case, r["yaw_track"] or 0,
                         r["support_tail"], r["upright_min"], r["terminations"]), flush=True)
            (args.out / "friction_turn.json").write_text(json.dumps(rep, indent=1, default=str))
            print("FRICTION_TURN_DONE", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
