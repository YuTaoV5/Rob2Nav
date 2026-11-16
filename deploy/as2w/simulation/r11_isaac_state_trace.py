#!/usr/bin/env python3
"""Dump a per-step closed-loop state trace from Isaac Lab, identical schema to r11_mj_state_trace.py.

The pair answers one question: given that the 65-dim observation is constructed identically on both sides
and the action chain is numerically consistent (C2 1.43e-06, C8 5.96e-07), WHICH FIELD of the closed-loop
state diverges first?  That field is the mechanism behind Isaac undershooting (0.05-0.31) while MuJoCo
overshoots (1.41-1.56) on the same weights.

Both traces record, per step: the 65-dim obs the policy saw, the 16-dim clipped action, measured q and dq
in contract order, root quaternion, root angular velocity in the WORLD frame, and yaw rate.
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
WHEEL_KD, ACTION_CLIP, DT = 6.0, 6.0, 0.02
JOINT_ORDER = [f"{leg}_{p}" for p in ("hip", "thigh", "calf", "foot") for leg in ("FL", "FR", "RL", "RR")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7393)
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.05)
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument("--turn-s", type=float, default=12.0)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=1)
    add_launcher_args(ap)
    args = ap.parse_args()
    for k in ("usd", "urdf", "ckpt"):
        setattr(args, k, getattr(args, k).resolve(strict=True))
    args.out.mkdir(parents=True, exist_ok=True)
    args.physics = "isaacsim_physx"

    with launch_simulation(SimulationCfg(device=args.device, dt=.005), args) as physics:
        # These imports MUST come AFTER the app is launched.  Importing the IsaacLab / rsl_rl stack
        # before launch_simulation loads a conflicting USD and the app dies with
        # "libusd_tf.so: undefined symbol pxrInternal_...".  All known-good probes import inside.
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
        cfg.commands.base_velocity.ranges.lin_vel_x = (-VX_CAP, VX_CAP)
        cfg.commands.base_velocity.ranges.lin_vel_y = (0., 0.)
        cfg.commands.base_velocity.ranges.ang_vel_z = (-YAW_CAP, YAW_CAP)
        cfg.episode_length_s = 60.
        cfg.scene.robot.actuators["foot"].damping = WHEEL_KD
        apply_safety(cfg, args.urdf, perturb=False, surface_guard=True)
        env = RobustEnvironment(cfg)
        try:
            wrapped = RslRlVecEnvWrapper(env, clip_actions=ACTION_CLIP)
            robot = env.scene["robot"]
            cmdterm = env.command_manager.get_term("base_velocity")
            obs = wrapped.get_observations()
            ppo = UnitreeGo2FlatPPORunnerCfg(); ppo.seed, ppo.device = args.seed, args.device
            ppo.max_iterations = 1
            ppo = handle_deprecated_rsl_rl_cfg(ppo, check_rsl_rl_version())
            runner = create_rsl_rl_runner(wrapped, ppo, log_dir=None)
            runner.load(str(args.ckpt), map_location=args.device)
            policy = runner.get_inference_policy(device=args.device)

            trace = {"sim": "isaac", "checkpoint": str(args.ckpt), "cmd": [args.vx, 0.0, args.wz],
                     "step_dt": DT, "settle_s": args.settle_s, "settle": [], "turn": []}

            wb_idx = [robot.body_names.index(n) for n in
                      [f"{L}_foot" for L in ("FL", "FR", "RL", "RR")]]

            def wheel_loads():
                try:
                    cf = env.scene["contact_forces"].data.net_forces_w.torch[0][wb_idx]
                    return [round(float(x), 2) for x in torch.linalg.norm(cf, dim=1)]
                except Exception:
                    return None

            def record(bucket, o, a):
                # get_observations() returns a TensorDict keyed by "policy"; o[0] would return the
                # KEY STRING rather than a tensor, which is what the first run tripped over.
                ovec = o["policy"] if hasattr(o, "keys") or not torch.is_tensor(o) else o
                trace_row = {
                    "obs": [round(float(x), 6) for x in ovec[0].detach().cpu().numpy()],
                    "act": [round(float(x), 6) for x in a[0].detach().cpu().numpy()],
                    "q": [round(float(x), 6) for x in robot.data.joint_pos.torch[0].detach().cpu().numpy()],
                    "dq": [round(float(x), 6) for x in robot.data.joint_vel.torch[0].detach().cpu().numpy()],
                    "quat": [round(float(x), 6) for x in robot.data.root_quat_w.torch[0].detach().cpu().numpy()],
                    "angvel": [round(float(x), 6) for x in robot.data.root_ang_vel_w.torch[0].detach().cpu().numpy()],
                    "yaw": round(float(robot.data.root_ang_vel_b.torch[0, 2]), 6),
                    "loads_n": wheel_loads()}
                bucket.append(trace_row)

            env.seed(args.seed); env.reset()
            cmdterm.vel_command_b[:] = 0.0
            obs = wrapped.get_observations()
            for _ in range(int(args.settle_s / DT)):
                with torch.inference_mode():
                    a = policy(obs)
                record(trace["settle"], obs, a)
                obs, _r, _d, _i = wrapped.step(a)
            cmdterm.vel_command_b[:] = torch.tensor([[args.vx, 0.0, args.wz]], dtype=torch.float32, device=env.device)
            for _ in range(int(args.turn_s / DT)):
                with torch.inference_mode():
                    a = policy(obs)
                record(trace["turn"], obs, a)
                obs, _r, _d, _i = wrapped.step(a)

            (args.out / "state_trace.json").write_text(json.dumps(trace))
            f = trace["turn"][len(trace["turn"]) // 2]
            print("settle frames %d, turn frames %d" % (len(trace["settle"]), len(trace["turn"])), flush=True)
            print("mid-turn yaw=%.5f  wheel dq=%s" % (f["yaw"], [round(x, 2) for x in f["dq"][12:]]), flush=True)
            print("ISAAC_STATE_TRACE_DONE", flush=True)
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
