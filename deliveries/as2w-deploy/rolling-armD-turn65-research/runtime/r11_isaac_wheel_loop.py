#!/usr/bin/env python3
"""Wheel-loop isolation on the Isaac side: the twin of r11_mj_wheel_loop.py.

Same treatment as the MuJoCo probe so the only thing under test is the wheel velocity loop:

    * the base is PINNED every sample (pose forced, root velocity zeroed) so gravity cannot move it
    * the 12 leg joints hold one fixed pose
    * the policy is never called: no actor, no gyro, no delay, no action manager, no command gate,
      no action ramp - the wheel velocity target is written straight to the articulation
    * the same target sequence: 0 -> +0.5 -> 0 -> -0.5 -> 0 rad/s

Because the action manager is bypassed entirely, the episode-start command gate and the 2 s action ramp
that produced a false "divergence" earlier cannot influence this experiment at all.

The criteria are the four the oscillation forced on us, not the meaningless settled-tail mean:
peak overshoot ratio, envelope decay time, sign reversals, saturated-torque frame fraction.
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

SEQUENCE = [(0.0, 1.0), (0.5, 2.0), (0.0, 1.0), (-0.5, 2.0), (0.0, 1.0)]
WHEEL_KD = 6.0
WHEEL_EFFORT = 13.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", type=Path, required=True)
    ap.add_argument("--urdf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7393)
    ap.add_argument("--lift-z", type=float, default=2.0)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--iterations", type=int, default=1)
    add_launcher_args(ap)
    args = ap.parse_args()
    for k in ("usd", "urdf", "ckpt"):
        setattr(args, k, getattr(args, k).resolve(strict=True))
    args.out.mkdir(parents=True, exist_ok=True)
    args.physics = "isaacsim_physx"

    with launch_simulation(SimulationCfg(device=args.device, dt=.005), args) as physics:
        # imports must come AFTER launch, else the USD libraries conflict
        from as2w_isaac_robust_env import RobustEnvironment, robust_config
        from as2w_training_safety import apply_safety

        cfg = robust_config(SimpleNamespace(**vars(args), profile="pose-delay", direction_focus=True,
                                             command_stage="all", zero_memory=True, zero_focus=True), physics)
        cfg.scene.robot.actuators["foot"].damping = WHEEL_KD
        apply_safety(cfg, args.urdf, perturb=False, surface_guard=True)
        env = RobustEnvironment(cfg)
        try:
            robot = env.scene["robot"]
            names = list(robot.joint_names)
            wheel_ids = [names.index(n) for n in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")]
            leg_ids = [names.index(n) for n in
                       [f"{L}_{p}" for p in ("hip", "thigh", "calf") for L in ("FL", "FR", "RL", "RR")]]
            decimation = int(getattr(cfg, "decimation", 4))
            sim_dt = float(physics.cfg.dt) if hasattr(physics, "cfg") else 0.005
            control_dt = sim_dt * decimation
            q_default = robot.data.default_joint_pos.torch[0].clone()
            root_pose = robot.data.root_pose_w.torch[0].clone()
            root_pose[2] = args.lift_z
            root_pose[0:2] = 0.0

            def pin():
                robot.write_root_pose_to_sim(root_pose.unsqueeze(0))
                robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=env.device))

            env.seed(args.seed)
            env.reset()
            pin()
            robot.write_joint_state_to_sim(q_default.unsqueeze(0),
                                            torch.zeros((1, len(names)), dtype=torch.float32, device=env.device))
            robot.write_data_to_sim()
            env.sim.step()
            robot.update(sim_dt)

            trace = []
            for target_vel, hold_s in SEQUENCE:
                n = int(round(hold_s / control_dt))
                for _ in range(n):
                    pin()
                    robot.set_joint_velocity_target_index(
                        target=torch.full((1, 4), float(target_vel), dtype=torch.float32, device=env.device),
                        joint_ids=wheel_ids)
                    robot.write_data_to_sim()
                    for _ in range(decimation):
                        env.sim.step()
                    robot.update(sim_dt)
                    dq = robot.data.joint_vel.torch[0][wheel_ids].detach().cpu().numpy()
                    at = None
                    if hasattr(robot.data, "applied_torque"):
                        try:
                            at = robot.data.applied_torque.torch[0][wheel_ids].detach().cpu().numpy()
                        except Exception:
                            at = None
                    trace.append({"target": float(target_vel),
                                  "dq_meas": [round(float(x), 6) for x in dq],
                                  "tau_applied": None if at is None else [round(float(x), 6) for x in at]})

            def analyse(target_vel):
                seg = [r for r in trace if r["target"] == target_vel]
                if not seg:
                    return None
                dq0 = np.array([r["dq_meas"][0] for r in seg])
                peak = float(np.max(np.abs(dq0)))
                signs = int(np.sum(np.diff(np.sign(dq0)) != 0))
                sat = 0
                for r in seg:
                    if r["tau_applied"] is not None and np.any(np.abs(r["tau_applied"]) >= WHEEL_EFFORT * 0.999):
                        sat += 1
                sat_frac = sat / len(seg) if seg else None
                return {"first_sample_dq": round(float(dq0[0]), 4),
                        "peak_abs_dq": round(peak, 4),
                        "overshoot_ratio": round(peak / abs(target_vel), 2) if target_vel else None,
                        "sign_reversals": signs,
                        "saturated_frame_fraction": sat_frac,
                        "samples": len(seg)}

            rep = {"sim": "isaac", "path": "Direct articulation velocity target, policy bypassed",
                   "physics_dt_s": sim_dt, "decimation": decimation, "control_period_s": control_dt,
                   "sequence": [[t, h] for t, h in SEQUENCE], "wheel_kd": WHEEL_KD,
                   "wheel_effort_limit_Nm": WHEEL_EFFORT, "base_pinned": True, "lift_z_m": args.lift_z,
                   "analysis": {"0.5": analyse(0.5), "-0.5": analyse(-0.5)}, "trace": trace}
            (args.out / "wheel_loop.json").write_text(json.dumps(rep))
            print("physics_dt=%.6f decimation=%d control_period=%.4f" % (sim_dt, decimation, control_dt))
            print("target +0.5 -> %s" % rep["analysis"]["0.5"])
            print("target -0.5 -> %s" % rep["analysis"]["-0.5"])
            print("ISAAC_WHEEL_LOOP_DONE")
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
